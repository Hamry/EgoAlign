"""Stage 2: embed frames in a pre-trained contrastive (CLIP) latent space."""

import cv2
import numpy as np


def thumbnail(rgb, size=224):
    """Downscales a frame so its short side is at most `size` px."""
    h, w = rgb.shape[:2]
    if min(h, w) <= size:
        return rgb
    scale = size / min(h, w)
    return cv2.resize(rgb, (round(w * scale), round(h * scale)),
                      interpolation=cv2.INTER_AREA)


class Encoder:
    """open_clip image encoder yielding L2-normalized, view-robust embeddings."""

    def __init__(self, model='ViT-B-32', pretrained='laion2b_s34b_b79k',
                 device=None, batch_size=32):
        """Loads the encoder; weights download on first use.

        Args:
            model: open_clip architecture name.
            pretrained: open_clip pretrained weights tag.
            device: Torch device; CUDA when available by default.
            batch_size: Frames per forward pass.

        Raises:
            ImportError: If the `align` extra is not installed.
        """
        # Optional heavy dependencies; imported only when an encoder is built.
        import open_clip  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
        self._torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model, pretrained=pretrained, device=self.device)
        self.model.eval()
        self.batch_size = batch_size

    def __call__(self, frames):
        """Embeds frames.

        Args:
            frames: Non-empty list of (H, W, 3) uint8 RGB frames.

        Returns:
            (N, D) float32 L2-normalized embeddings.

        Raises:
            ValueError: If `frames` is empty.
        """
        if not frames:
            raise ValueError('no frames to embed')
        from PIL import Image  # pylint: disable=import-outside-toplevel
        torch = self._torch
        out = []
        for i in range(0, len(frames), self.batch_size):
            batch = torch.stack([self.preprocess(Image.fromarray(f))
                                 for f in frames[i:i + self.batch_size]])
            with torch.no_grad():
                emb = self.model.encode_image(batch.to(self.device))
            out.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)
