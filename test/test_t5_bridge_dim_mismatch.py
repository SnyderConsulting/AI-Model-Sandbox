import importlib.util
import types
import torch
import torch.nn as nn
import pytest

T5_BRIDGE_PATH = (
    __import__("pathlib").Path(__file__).resolve().parents[1]
    / "inference"
    / "Wan2.2"
    / "wan"
    / "modules"
    / "t5_bridge.py"
)
spec = importlib.util.spec_from_file_location("t5_bridge", T5_BRIDGE_PATH)
t5_bridge = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(t5_bridge)
BridgeEncoderModel = t5_bridge.BridgeEncoderModel


class DummyLLM(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden_size)

    def forward(self, *args, **kwargs):  # pragma: no cover - not used
        return None


def test_bridge_dim_mismatch(tmp_path, monkeypatch):
    ckpt_path = tmp_path / "bridge.pth"
    torch.save({"bridge": {"in_proj.weight": torch.zeros(1024, 3584)}}, ckpt_path)

    dummy_llm = DummyLLM(hidden_size=5120)

    monkeypatch.setenv("WAN_BRIDGE_LLM_DIR", "dummy")
    monkeypatch.setenv("WAN_BRIDGE_CKPT", str(ckpt_path))
    monkeypatch.setenv("WAN_TEXT_LEN", "16")
    monkeypatch.setenv("WAN_TEXT_DIM", "8")

    monkeypatch.setattr(
        t5_bridge.AutoTokenizer,
        "from_pretrained",
        lambda *a, **k: object(),
    )
    monkeypatch.setattr(
        t5_bridge.AutoModelForCausalLM,
        "from_pretrained",
        lambda *a, **k: dummy_llm,
    )

    with pytest.raises(RuntimeError) as excinfo:
        BridgeEncoderModel(text_len=16, device="cpu")
    assert "Bridge/LLM hidden_size mismatch" in str(excinfo.value)
