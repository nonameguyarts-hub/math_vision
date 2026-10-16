"""recognizer.py - the real recognizer. The UI calls recognize(image) -> LaTeX string.

Put best.pt (from MathVision/finetune_reg/ on Drive) in  models/best.pt  next to this file,
or set the environment variable MATHVISION_CKPT to its path.
"""
import os
from pathlib import Path

from preprocess import preprocess

DEFAULT_CKPT = os.environ.get("MATHVISION_CKPT", str(Path(__file__).resolve().parent / "models" / "best.pt"))


class Recognizer:
    def __init__(self, ckpt_path=DEFAULT_CKPT, threads=None):
        path = Path(ckpt_path)
        if not path.exists():
            raise NotImplementedError(f"Model file not found at {path}.")
        import torch
        from model import Im2Latex
        try:
            ck = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:                                   # checkpoint holds extra training state
            ck = torch.load(path, map_location="cpu", weights_only=False)
        self.vocab = ck["config"]["vocab"]
        ids = {t: i for i, t in enumerate(self.vocab)}
        self.pad, self.sos, self.eos = ids["<pad>"], ids["<sos>"], ids["<eos>"]
        self.model = Im2Latex(len(self.vocab), self.pad, self.sos, self.eos)
        self.model.load_state_dict(ck["model"])             # strict: fails loudly if layers differ
        self.model.eval()
        if threads:
            torch.set_num_threads(threads)
        self.torch, self.epoch = torch, ck.get("epoch")

    def recognize(self, image, autocrop: bool = True, rotate: int = 0) -> str:
        canvas, w = preprocess(image, autocrop, rotate)
        x = self.torch.from_numpy(canvas)[None, None]
        widths = self.torch.tensor([w])
        with self.torch.inference_mode():
            ids = self.model.predict(x, widths)[0].tolist()
        out = []
        for i in ids:
            if i == self.eos:
                break
            if i not in (self.pad, self.sos):
                out.append(self.vocab[i])
        return " ".join(out)


_rec = None


def recognize(image, rotate: int = 0) -> str:
    """Load the model on first use, then reuse it."""
    global _rec
    if _rec is None:
        _rec = Recognizer()
    return _rec.recognize(image, rotate=rotate)