import os
from typing import Dict, Optional

import numpy as np

CAPTION_REPO = "Rohan3/ShapeNetCore_Captions"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


class ShapeNetCaptionIndex:
    """model_id -> caption map built from the gated HF captions CSV (6.79 MB)."""

    def __init__(self, caption_map: Dict[str, str]):
        self.caption_map = caption_map

    @classmethod
    def load(cls, token: Optional[str] = None, filename: Optional[str] = None) -> "ShapeNetCaptionIndex":
        import io

        import pandas as pd
        from huggingface_hub import hf_hub_download, list_repo_files

        token = token or os.environ.get("HF_TOKEN")
        if filename is None:
            files = [f for f in list_repo_files(CAPTION_REPO, repo_type="dataset", token=token) if f.endswith(".csv")]
            assert files, "no CSV found in captions repo"
            filename = sorted(files)[0]
        path = hf_hub_download(CAPTION_REPO, repo_type="dataset", filename=filename, token=token)
        df = pd.read_csv(io.BytesIO(open(path, "rb").read()))
        id_col = next(c for c in df.columns if "id" in c.lower() or "model" in c.lower())
        cap_col = next(c for c in df.columns if "caption" in c.lower() or "text" in c.lower() or "desc" in c.lower())
        caption_map = {str(mid): str(cap) for mid, cap in zip(df[id_col], df[cap_col])}
        return cls(caption_map)

    def __getitem__(self, model_id: str) -> str:
        return self.caption_map[model_id]

    def __contains__(self, model_id: str) -> bool:
        return model_id in self.caption_map

    def __len__(self) -> int:
        return len(self.caption_map)


def build_clip_embedding_cache(
    captions: Dict[str, str],
    out_dir: str,
    model_name: str = DEFAULT_CLIP_MODEL,
    batch_size: int = 256,
    device: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Encodes all captions once with a frozen CLIP text encoder.

    Saves embeddings.npy (N,512 fp16) + ids.json; returns the id->row index map.
    """
    import json

    import torch
    from transformers import CLIPTextModel, CLIPTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPTextModel.from_pretrained(model_name).to(device).eval()

    ids = sorted(captions.keys())
    embs = np.zeros((len(ids), model.config.hidden_size), dtype=np.float16)
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start : start + batch_size]
            texts = [captions[i] for i in batch_ids]
            tokens = tokenizer(texts, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
            out = model(input_ids=tokens["input_ids"].to(device))
            embs[start : start + len(batch_ids)] = out.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float16)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "caption_embeddings.npy"), embs)
    with open(os.path.join(out_dir, "caption_ids.json"), "w") as f:
        json.dump({mid: i for i, mid in enumerate(ids)}, f)
    return {mid: i for i, mid in enumerate(ids)}
