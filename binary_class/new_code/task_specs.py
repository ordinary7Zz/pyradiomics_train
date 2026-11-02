from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class AgTaskSpec:
    name: str
    label_key: str
    packed_root: Path
    train_manifest: str
    image_roots_json: str
    zjh_mask_root: Path
    lymphus_mask_root: Optional[Path]
    fangdai_mask_root: Optional[Path]
    # 默认 AG（与 my_run.md 恶性任务一致，可按任务覆盖）
    resample_strategy: str
    target_class0_count: int
    target_class1_count: int


LNM_PACK = Path(
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/LNM_ZJH8K_LymphUs_C1C2_packed"
)
FTC_PACK = Path(
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/FTCPTC_ZJH8K_FangDai_packed"
)
ZJH_MASK = Path(
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped_predictions"
)
LYMPHUS_MASK = Path(
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/pred_masks/Lymph_Node_Metastasis_fake/Lymph_images"
)
FANGDAI_MASK = Path(
    "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/FangDai_Thyroid_Ultrasound_Images_cropped_predictions"
)

TASKS = {
    "lnm": AgTaskSpec(
        name="lnm",
        label_key="LNM_CN01",
        packed_root=LNM_PACK,
        train_manifest="manifests/train_combined_zjh_plus_lymphus_c1.json",
        image_roots_json="image_roots.json",
        zjh_mask_root=ZJH_MASK,
        lymphus_mask_root=LYMPHUS_MASK,
        fangdai_mask_root=None,
        resample_strategy="none",
        target_class0_count=-1,
        target_class1_count=-1,
    ),
    "ftcptc": AgTaskSpec(
        name="ftcptc",
        label_key="FTCPTC",
        packed_root=FTC_PACK,
        train_manifest="manifests/zjh_dev_all.json",
        image_roots_json="image_roots.json",
        zjh_mask_root=ZJH_MASK,
        lymphus_mask_root=None,
        fangdai_mask_root=FANGDAI_MASK,
        resample_strategy="none",
        target_class0_count=-1,
        target_class1_count=-1,
    ),
}


def get_task(name: str) -> AgTaskSpec:
    key = name.lower().strip()
    if key not in TASKS:
        raise KeyError(f"Unknown task {name!r}; choose from {list(TASKS)}")
    return TASKS[key]
