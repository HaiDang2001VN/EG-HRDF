"""Semantic alignment metrics: text-view CLIP score (S_TV) and cross-view consistency (S_CV)."""

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .rendering import render_point_cloud_views

DEFAULT_CLIP = "openai/clip-vit-base-patch32"


class CLIPScorer:
    def __init__(self, model_name: str = DEFAULT_CLIP, device: str = "cpu"):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor, batch_size: int = 16) -> torch.Tensor:
        """images: (V, 3, H, W) in [0,1] -> (V, D) normalized embeddings."""
        embs = []
        for i in range(0, images.shape[0], batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            out = self.model.get_image_features(
                pixel_values=inputs["pixel_values"].to(self.device)
            )
            embs.append(F.normalize(out, dim=-1).cpu())
        return torch.cat(embs, dim=0)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        out = self.model.get_text_features(
            input_ids=inputs["input_ids"].to(self.device),
            attention_mask=inputs["attention_mask"].to(self.device),
        )
        return F.normalize(out, dim=-1).cpu()


def semantic_scores(
    points: np.ndarray,
    caption: str,
    scorer: CLIPScorer,
    image_size: int = 224,
) -> Dict[str, float]:
    """S_TV: mean cosine(text, view) over fixed views; S_CV: mean pairwise view-view cosine."""
    views = render_point_cloud_views(points, image_size=image_size)
    img_emb = scorer.encode_images(views)
    txt_emb = scorer.encode_texts([caption])
    s_tv = (img_emb @ txt_emb.t()).mean().item()
    n = img_emb.shape[0]
    if n > 1:
        sim = img_emb @ img_emb.t()
        mask = ~torch.eye(n, dtype=torch.bool)
        s_cv = sim[mask].mean().item()
    else:
        s_cv = 1.0
    return {"S_TV": s_tv, "S_CV": s_cv}
