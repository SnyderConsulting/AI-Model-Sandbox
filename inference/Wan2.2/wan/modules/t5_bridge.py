# wan/modules/t5_bridge.py
# MythoMax/LLaMA → Wan-space bridge as a drop-in replacement for Wan's T5EncoderModel.

import os
import math
from typing import List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:
    Qwen2_5_VLForConditionalGeneration = None


def _sanitize_texts(texts):
    clean = []
    for t in texts:
        if not isinstance(t, str):
            t = str(t)
        t = (
            t.encode("utf-8", "ignore")
            .decode("utf-8", "ignore")
            .replace("\x00", "")
            .strip()
        )
        clean.append(t)
    return clean


# ------------------------ utils ------------------------


def _resolve_device(
    device_id: Union[int, str, torch.device, None], t5_cpu: bool
) -> torch.device:
    """
    device_id can be int (local rank), str ('cuda', 'cuda:0', 'cpu'), torch.device, or None.
    t5_cpu forces CPU regardless of device_id.
    """
    if t5_cpu:
        return torch.device("cpu")
    if isinstance(device_id, torch.device):
        return device_id
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if isinstance(device_id, int):
        return torch.device(f"cuda:{device_id}")
    if isinstance(device_id, str):
        s = device_id.strip().lower()
        if s == "cpu":
            return torch.device("cpu")
        if s == "cuda":
            return torch.device("cuda:0")
        if s.startswith("cuda:"):
            return torch.device(s)
    # Fallbacks
    try:
        return torch.device("cuda:0")
    except Exception:
        return torch.device("cpu")


# ------------------------ bridge (Perceiver-style) ------------------------


class _MLP(nn.Module):
    def __init__(self, d: int, mult: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(d, mult * d)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mult * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _CrossBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.q_ln = nn.LayerNorm(d, eps=1e-5)
        self.kv_ln = nn.LayerNorm(d, eps=1e-5)
        self.mha = nn.MultiheadAttention(
            embed_dim=d, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.mlp_ln = nn.LayerNorm(d, eps=1e-5)
        self.mlp = _MLP(d, mult=4)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        qn, kvn = self.q_ln(q), self.kv_ln(kv)
        attn_out, _ = self.mha(
            qn,
            kvn,
            kvn,
            key_padding_mask=(~kv_mask) if kv_mask is not None else None,
            need_weights=False,
        )
        x = q + attn_out
        x = x + self.mlp(self.mlp_ln(x))
        return x


class PerceiverBridge(nn.Module):
    """
    Map LLM token states [B, Lt, d_llm] → Wan-style tokens [B, L_wan, d_wan].
    """

    def __init__(
        self,
        d_llm: int = 5120,
        d_wan: int = 4096,
        L_wan: int = 512,
        d_mid: int = 1024,
        n_heads: int = 16,
        n_blocks: int = 3,
    ):
        super().__init__()
        self.L_wan = L_wan
        self.query = nn.Parameter(torch.randn(L_wan, d_mid) / math.sqrt(d_mid))
        self.in_proj = nn.Linear(d_llm, d_mid)
        self.blocks = nn.ModuleList(
            [_CrossBlock(d_mid, n_heads) for _ in range(n_blocks)]
        )
        self.out_ln = nn.LayerNorm(d_mid, eps=1e-5)
        self.out_proj = nn.Linear(d_mid, d_wan)
        # learned affine to match Wan feature stats
        self.out_scale = nn.Parameter(torch.ones(1, 1, d_wan))
        self.out_shift = nn.Parameter(torch.zeros(1, 1, d_wan))

    def forward(
        self, llm_tokens: torch.Tensor, llm_mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        B, Lt, _ = llm_tokens.shape
        x_tokens = self.in_proj(llm_tokens)
        q = self.query.unsqueeze(0).expand(B, -1, -1).contiguous()
        for blk in self.blocks:
            q = blk(q, x_tokens, kv_mask=llm_mask)
        h = self.out_proj(self.out_ln(q))
        return h * self.out_scale + self.out_shift  # [B, L_wan, d_wan]


# ------------------------ drop-in encoder ------------------------


class BridgeEncoderModel:
    """
    Drop-in replacement for Wan's T5EncoderModel.

    __call__(texts, device) -> List[Tensor[L_i<=L_wan, d_wan]]

    Env:
      WAN_BRIDGE_LLM_DIR   : path to HF LLM (e.g., MythoMax-L2-13B)
      WAN_BRIDGE_CKPT      : path to trained bridge checkpoint (*.pth with {"bridge": ...})
      WAN_TEXT_LEN         : default 512
      WAN_TEXT_DIM         : default 4096 (TI2V-5B)
      WAN_BRIDGE_DTYPE     : bf16|fp16 (default bf16)
      WAN_LLM_MAXLEN       : default 512
      WAN_BRIDGE_GLOBAL_SCALE : optional global scale applied to tokens (default 1.0)
      WAN_BRIDGE_GLOBAL_BIAS  : optional global bias added to tokens (default 0.0)
    """

    def __init__(
        self,
        text_len: int,
        dtype: torch.dtype = torch.bfloat16,
        device: Union[int, str, torch.device] = "cuda",
        checkpoint_path: Optional[str] = None,  # ignored (we use env)
        tokenizer_path: Optional[str] = None,  # ignored
        shard_fn=None,  # ignored (Accelerate handles sharding)
        **kwargs,  # absorb unknown args to stay API-compatible
    ):
        # --- config / env ---
        self.L_wan = int(os.environ.get("WAN_TEXT_LEN", text_len or 512))
        self.d_wan = int(os.environ.get("WAN_TEXT_DIM", 4096))
        self.llm_dir = os.environ.get(
            "WAN_BRIDGE_LLM_DIR", "/workspace/models/MythoMax-L2-13B"
        )
        self.ckpt_path = os.environ.get("WAN_BRIDGE_CKPT", "")
        self.llm_max = int(os.environ.get("WAN_LLM_MAXLEN", 512))
        dt_str = os.environ.get("WAN_BRIDGE_DTYPE", "bf16").lower()
        self.dtype = torch.bfloat16 if dt_str in ("bf16", "bfloat16") else torch.float16
        t5_force_cpu = bool(kwargs.get("t5_cpu", False))
        self.device = _resolve_device(device, t5_force_cpu)

        # NEW: global affine knobs for crude calibration (env-driven)
        self.global_scale = float(os.environ.get("WAN_BRIDGE_GLOBAL_SCALE", "1.0"))
        self.global_bias = float(os.environ.get("WAN_BRIDGE_GLOBAL_BIAS", "0.0"))

        if not self.ckpt_path or not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(
                "WAN_BRIDGE_CKPT must point to a valid bridge checkpoint (*.pth)."
            )

        # --- load LLM (tokenizer/processor + model) ---
        self.is_vl = False
        device_map = "auto" if self.device.type == "cuda" else {"": "cpu"}
        force_vl = os.environ.get("WAN_BRIDGE_FORCE_VL", "1").lower() not in (
            "0",
            "false",
            "no",
        )

        if Qwen2_5_VLForConditionalGeneration is not None:
            try:
                self.proc = AutoProcessor.from_pretrained(
                    self.llm_dir, trust_remote_code=True
                )
                self.llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.llm_dir,
                    torch_dtype=self.dtype,
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                self.is_vl = True
            except Exception as e:
                if force_vl:
                    print(
                        "[BridgeEncoderModel] Qwen2.5-VL load failed; refusing to fall back because "
                        "WAN_BRIDGE_FORCE_VL is set. Full error:"
                    )
                    raise
                else:
                    print(
                        f"[BridgeEncoderModel] Qwen2.5-VL load failed ({e}). Falling back to AutoModelForCausalLM."
                    )

        if not self.is_vl:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tok = AutoTokenizer.from_pretrained(
                self.llm_dir, use_fast=True, trust_remote_code=True
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.llm_dir,
                torch_dtype=self.dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

        self.llm.eval().requires_grad_(False)

        d_llm = getattr(self.llm.config, "hidden_size", None)
        if d_llm is None:
            raise RuntimeError(
                "Loaded LLM has no config.hidden_size; cannot build bridge."
            )

        # --- build + load bridge ---
        self.bridge = (
            PerceiverBridge(
                d_llm=d_llm,
                d_wan=self.d_wan,
                L_wan=self.L_wan,
                d_mid=1024,
                n_heads=16,
                n_blocks=3,
            )
            .to(self.device, dtype=self.dtype)
            .eval()
        )

        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        sd = ckpt.get("bridge", ckpt)

        # If the bridge was saved as raw module weights, pick a canonical key:
        in_keys = [k for k in sd.keys() if k.endswith("in_proj.weight")]
        if in_keys:
            ckpt_d_llm = sd[in_keys[0]].shape[1]
            if ckpt_d_llm != d_llm:
                raise RuntimeError(
                    f"Bridge/LLM hidden_size mismatch: bridge expects d_llm={ckpt_d_llm}, "
                    f"but loaded LLM has d_llm={d_llm}. "
                    "This is almost always due to falling back to a CausalLM or pointing to the wrong LLM repo."
                )
        missing, unexpected = self.bridge.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(
                f"[BridgeEncoderModel] Warning while loading state_dict: missing={missing}, unexpected={unexpected}"
            )

        # --- perf knobs ---
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        # --- expose .model so Wan can call text_encoder.model.to(...) ---
        outer = self

        class _BridgeContainer(nn.Module):
            def __init__(self, llm: nn.Module, bridge: nn.Module):
                super().__init__()
                self.llm = llm
                self.bridge = bridge

            def to(self, *args, **kwargs):
                # Allow .to("cuda:0") and/or .to(dtype=torch.bfloat16)
                device = kwargs.get("device", None)
                dtype = kwargs.get("dtype", None)
                if args:
                    # positional device/dtype
                    if isinstance(args[0], (torch.device, str, int)):
                        device = args[0]
                    elif isinstance(args[0], torch.dtype):
                        dtype = args[0]
                # Move bridge always; move LLM only if not sharded by Accelerate
                if device is not None:
                    if not hasattr(self.llm, "hf_device_map"):
                        self.llm.to(device)
                    self.bridge.to(device)
                    # update the outer hint to the *current* LLM param device
                    try:
                        outer_llm_dev = next(self.llm.parameters()).device
                    except StopIteration:
                        outer_llm_dev = (
                            torch.device(device)
                            if not isinstance(device, torch.device)
                            else device
                        )
                    outer._llm_in_device = (
                        outer_llm_dev  # used for token tensor placement
                    )
                if dtype is not None:
                    if not hasattr(self.llm, "hf_device_map"):
                        self.llm.to(dtype=dtype)
                    self.bridge.to(dtype=dtype)
                return self

        self.model = _BridgeContainer(self.llm, self.bridge)

        # initial hint (may change when Wan calls .model.to(...))
        try:
            self._llm_in_device = next(self.llm.parameters()).device
        except StopIteration:
            self._llm_in_device = self.device

        print(
            f"[BridgeEncoderModel] LLM={self.llm_dir} d_llm={d_llm}  _  Wan[L={self.L_wan}, D={self.d_wan}] "
            f"ckpt={os.path.basename(self.ckpt_path)}  dtype={self.dtype} device={self.device} "
            f"gscale={self.global_scale} gbias={self.global_bias}"
        )

    @torch.no_grad()
    def __call__(
        self, texts: List[str], device: Optional[Union[int, str, torch.device]] = None
    ):
        """
        Returns a list of [L_i, d_wan] tensors (trimmed per-sample lengths),
        mirroring Wan's stock T5EncoderModel API.
        """
        out_device = (
            _resolve_device(device, False) if device is not None else self.device
        )
        texts = _sanitize_texts(texts)

        try:
            llm_in_device = next(self.llm.parameters()).device
        except StopIteration:
            llm_in_device = self._llm_in_device

        inputs = None
        enc = None
        if self.is_vl:
            inputs = self.proc(
                text=texts,
                images=None,
                videos=None,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.llm_max,
            )
            inputs = {
                k: (v.to(llm_in_device) if hasattr(v, "to") else v)
                for k, v in inputs.items()
            }
            out = self.llm(**inputs, output_hidden_states=True, use_cache=False)
        else:
            enc = self.tok(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.llm_max,
            ).to(llm_in_device)
            out = self.llm(**enc, output_hidden_states=True, use_cache=False)

        h_llm = out.hidden_states[-1]
        m_llm = (
            inputs["attention_mask"] if self.is_vl else enc["attention_mask"]
        ).bool()

        # Run the bridge on *its current* device
        try:
            bridge_device = next(self.bridge.parameters()).device
        except StopIteration:
            bridge_device = out_device

        H_wan = self.bridge(
            h_llm.to(bridge_device), m_llm.to(bridge_device)
        )  # [B, L_wan, d_wan]

        # --- crude global affine for sanity test ---
        if self.global_scale != 1.0 or self.global_bias != 0.0:
            H_wan = H_wan * self.global_scale + self.global_bias

        H_wan = H_wan.to(out_device)

        # Trim each sample to its effective token length (≤ L_wan)
        lengths = m_llm.sum(dim=1).clamp(min=1, max=self.L_wan).tolist()
        out_list: List[torch.Tensor] = []
        for i, L in enumerate(lengths):
            out_list.append(H_wan[i, :L, :].contiguous())
        return out_list
