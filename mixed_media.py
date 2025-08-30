from __future__ import annotations
import os, re, math, random, pathlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset, Sampler
from torchvision.io import read_image, ImageReadMode, read_video
import torchvision.transforms.functional as TF

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}
VID_EXT = {".mp4", ".webm", ".mkv", ".mov"}


@dataclass
class Sample:
    path: str
    txt: str
    is_video: bool
    res: int  # 480 or 720
    frames: int  # 1 for images; else 17/33/49/65/81
    bucket: Tuple[int, int]  # (res, frames)


def _caption_path(stem: str) -> str:
    alt = stem + ".txt"
    return alt if os.path.exists(alt) else ""


def _infer_frames_from_path(path: str, default_frames: List[int]) -> int:
    # Expect folder names like ".../17frames", ".../33frames"
    m = re.search(r"(\d+)frames", path.replace("\\", "/"))
    if m:
        return int(m.group(1))
    return default_frames[-1]  # fallback for videos placed directly in 480p/720p


def _norm_to_range(x: torch.Tensor) -> torch.Tensor:
    # uint8 -> float32 in [-1,1]
    if x.dtype != torch.float32:
        x = x.float()
    return (x / 255.0) * 2.0 - 1.0


class MixedCaptioned(Dataset):
    """
    Recursively scans:
      /data/image/{480p,720p}/*.png + .txt pairs
      /data/video/{480p,720p}/{17frames,33frames,...}/*.mp4 + .txt pairs
    """

    def __init__(
        self,
        root: str,
        frames_options: List[int],  # e.g. [1,17] or [1,17,33,49,65,81]
        resolutions: List[int],  # e.g. [480,720]
        center_crop: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        assert 1 in frames_options, "Include 1 in --frames so images are supported."
        self.root = pathlib.Path(root)
        self.frames_options = sorted(frames_options)
        self.resolutions = sorted(resolutions)
        self.center_crop = center_crop
        self.rng = random.Random(seed)

        self.samples: List[Sample] = []
        for part in ["image", "video"]:
            for res in self.resolutions:
                base = self.root / part / f"{res}p"
                if not base.exists():
                    continue
                if part == "image":
                    for p in base.rglob("*"):
                        if p.suffix.lower() in IMG_EXT:
                            txt = _caption_path(p.with_suffix("").as_posix())
                            if not txt:
                                continue
                            self.samples.append(
                                Sample(
                                    path=p.as_posix(),
                                    txt=txt,
                                    is_video=False,
                                    res=res,
                                    frames=1,
                                    bucket=(res, 1),
                                )
                            )
                else:
                    # video case: look under Nframes folders
                    for p in base.rglob("*"):
                        if p.suffix.lower() in VID_EXT:
                            txt = _caption_path(p.with_suffix("").as_posix())
                            if not txt:
                                continue
                            nfrm = _infer_frames_from_path(
                                p.as_posix(), self.frames_options
                            )
                            if nfrm not in self.frames_options:
                                # skip videos with a frame count we didn't ask to train
                                continue
                            self.samples.append(
                                Sample(
                                    path=p.as_posix(),
                                    txt=txt,
                                    is_video=True,
                                    res=res,
                                    frames=nfrm,
                                    bucket=(res, nfrm),
                                )
                            )
        if not self.samples:
            raise RuntimeError(
                f"No (image|video)+caption pairs found under {self.root}"
            )

        # Buckets -> indices
        self.buckets: Dict[Tuple[int, int], List[int]] = {}
        for i, s in enumerate(self.samples):
            self.buckets.setdefault(s.bucket, []).append(i)
        for k in self.buckets:
            self.rng.shuffle(self.buckets[k])

    def __len__(self):
        return len(self.samples)

    def _resize_and_crop(self, x: torch.Tensor, target: int) -> torch.Tensor:
        # x: [C,H,W], uint8 or float; output float32 in [-1,1]; square or keep aspect, shortest side -> target
        C, H, W = x.shape
        # make shortest side == target, keep aspect
        scale = target / min(H, W)
        newH = int(round(H * scale))
        newW = int(round(W * scale))
        x = TF.resize(x, [newH, newW], antialias=True)
        # center/ random crop to (target,target) if needed
        if newH != target or newW != target:
            top = (
                (newH - target) // 2
                if self.center_crop
                else self.rng.randint(0, newH - target)
            )
            left = (
                (newW - target) // 2
                if self.center_crop
                else self.rng.randint(0, newW - target)
            )
            x = TF.crop(x, top, left, target, target)
        return _norm_to_range(x)

    def _load_image_tensor(self, path: str, target: int) -> torch.Tensor:
        # returns [T=1, 3, H, W] normalized [-1,1]
        im = read_image(path, mode=ImageReadMode.RGB)  # [3,H,W], uint8
        im = self._resize_and_crop(im, target)
        return im.unsqueeze(0)  # [1,3,H,W]

    def _load_video_tensor(self, path: str, target: int, nframes: int) -> torch.Tensor:
        # decode then uniform-sample nframes
        # read_video returns (video[T,H,W,C], audio, info); video is float32 0..255
        v, _, info = read_video(path, pts_unit="sec")
        if v.ndim != 4:
            raise RuntimeError(f"Bad video tensor for {path}: {v.shape}")
        total = v.shape[0]
        if total < nframes:
            # loop-pad
            reps = math.ceil(nframes / total)
            v = v.repeat(reps, 1, 1, 1)[:nframes]
        else:
            idx = torch.linspace(0, total - 1, nframes).round().long()
            v = v.index_select(0, idx)

        # [T,H,W,C] -> list of [3,H,W] tensors
        frames = v.permute(0, 3, 1, 2).contiguous()  # [T,3,H,W], 0..255 float
        out = []
        for i in range(frames.size(0)):
            out.append(self._resize_and_crop(frames[i], target))  # [3,h,w]
        return torch.stack(out, dim=0)  # [T,3,h,w] in [-1,1]

    def __getitem__(self, idx):
        s = self.samples[idx]
        with open(s.txt, "r", encoding="utf-8") as f:
            caption = f.read().strip()

        if s.is_video:
            x = self._load_video_tensor(s.path, s.res, s.frames)  # [T,3,h,w]
        else:
            x = self._load_image_tensor(s.path, s.res)  # [1,3,h,w]
        return dict(
            pixel=x,  # normalized [-1,1]
            caption=caption,
            res=s.res,
            frames=s.frames,
            is_video=s.is_video,
        )


class BucketBatchSampler(Sampler[List[int]]):
    """
    Yields index lists such that each batch has the same (res,frames) bucket.
    You pass a map bucket->indices and a per-bucket batch size table.
    """

    def __init__(
        self,
        dataset: MixedCaptioned,
        batch_sizes: Dict[Tuple[int, int], int],
        seed: int = 0,
    ):
        self.ds = dataset
        self.batch_sizes = batch_sizes
        self.rng = random.Random(seed)
        # clone + shuffle pointers for each bucket
        self.ptrs: Dict[Tuple[int, int], int] = {k: 0 for k in self.ds.buckets}

    def __iter__(self):
        # cycle buckets round-robin
        keys = list(self.ds.buckets.keys())
        self.rng.shuffle(keys)
        while True:
            made_any = False
            for k in keys:
                inds = self.ds.buckets[k]
                bs = self.batch_sizes.get(k, 1)
                p = self.ptrs[k]
                if p >= len(inds):
                    continue
                made_any = True
                q = min(p + bs, len(inds))
                self.ptrs[k] = q
                yield inds[p:q]
            if not made_any:
                break

    def __len__(self):
        # approximate (number of batches across all buckets)
        total = 0
        for k, inds in self.ds.buckets.items():
            bs = self.batch_sizes.get(k, 1)
            total += math.ceil(len(inds) / bs)
        return total
