import json
from dataclasses import dataclass, field
from typing import List, Optional

from huggingface_hub import hf_hub_download

from .caption_index import ShapeNetCaptionIndex

LABELS_FILE = "meta/all/labels.json"


@dataclass
class JoinReport:
    n_geometry: int
    n_matched: int
    n_missing: int
    missing_ids: List[str] = field(default_factory=list)

    @property
    def match_rate(self) -> float:
        return self.n_matched / max(self.n_geometry, 1)

    def summary(self) -> str:
        return (
            f"geometry={self.n_geometry} matched={self.n_matched} missing={self.n_missing} "
            f"rate={self.match_rate:.4f}"
        )


def load_category_ids(category: Optional[str] = None, split: Optional[str] = None) -> List[str]:
    path = hf_hub_download("EPFL-IVRL/ShapeNetSDF", repo_type="dataset", filename=LABELS_FILE)
    labels = json.load(open(path))
    if category is None:
        ids = []
        for v in labels["category_to_filename"].values():
            ids.extend(v)
        return ids
    return list(labels["category_to_filename"][category])


def validate_join(
    captions: ShapeNetCaptionIndex,
    object_ids: List[str],
    min_rate: float = 0.99,
) -> JoinReport:
    missing = [mid for mid in object_ids if mid not in captions]
    matched = len(object_ids) - len(missing)
    report = JoinReport(
        n_geometry=len(object_ids),
        n_matched=matched,
        n_missing=len(missing),
        missing_ids=missing[:100],
    )
    if report.match_rate < min_rate:
        raise ValueError(
            f"join rate {report.match_rate:.4f} below required {min_rate}; "
            f"first missing ids: {report.missing_ids[:10]}"
        )
    return report
