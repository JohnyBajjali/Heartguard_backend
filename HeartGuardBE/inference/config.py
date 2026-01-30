from dataclasses import dataclass
from django.conf import settings

@dataclass(frozen=True)
class NnUNetConfig:
    results_dir: str = settings.NNUNET_RESULTS_DIR
    checkpoint: str = settings.NNUNET_CHECKPOINT
    cfg: str = settings.NNUNET_CFG
    folds: list[str] = settings.NNUNET_FOLDS

    ds_3x4plus1: str = settings.NNUNET_DATASET_3X4PLUS1
    ds_3x4: str = settings.NNUNET_DATASET_3X4
