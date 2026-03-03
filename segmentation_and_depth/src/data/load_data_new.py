import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset


# ---------------------------------------------------------------------------
# map name -> channel count
# ---------------------------------------------------------------------------

MAPS_AND_DEPTHS = {
    "VesselMask": 1,
    "VesselOpening_Depth": 1,
    "VesselWithContentRGB": 3,
    "VesselWithContentNormal": 3,
    "VesselWithContentDepth": 1,
    "EmptyVessel_Depth": 1,
    "ContentNormal": 3,
    "ContentDepth": 1,
    "ContentMask": 3,
    "ContentMaskClean": 1,
    "VesselOpeningMask": 1,
    "ROI": 1,
}

MAPS_AND_DEPTHS_LABPICS = {
    "VesselMask": 1,
    "VesselWithContentRGB": 3,
    "ContentMaskClean": 1,
    "ROI": 1,
}


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _crop_resize(maps: dict, hb: int, wb: int) -> dict:
    """crop and resize all array-valued maps to (hb, wb)."""
    h, w = maps["ROI"].shape
    bs = np.min((h / hb, w / wb))
    if bs < 1 or bs > 3 or np.random.rand() < 0.2:
        h = int(h / bs) + 1
        w = int(w / bs) + 1
        for nm, arr in maps.items():
            if isinstance(arr, np.ndarray):
                interp = cv2.INTER_LINEAR if "RGB" in nm else cv2.INTER_NEAREST
                maps[nm] = cv2.resize(arr, dsize=(w, h), interpolation=interp)

    x0 = np.random.randint(w - wb) if w > wb else 0
    y0 = np.random.randint(h - hb) if h > hb else 0

    for nm, arr in maps.items():
        if isinstance(arr, np.ndarray):
            maps[nm] = arr[y0: y0 + hb, x0: x0 + wb]

    # safety resize if still wrong shape
    for nm, arr in maps.items():
        if isinstance(arr, np.ndarray):
            if arr.shape[0] != hb or arr.shape[1] != wb:
                maps[nm] = cv2.resize(arr, dsize=(wb, hb), interpolation=cv2.INTER_NEAREST)

    return maps


def _augment(maps: dict) -> dict:
    """per-sample augmentations applied to RGB maps."""
    for nm, arr in maps.items():
        if "RGB" not in nm or not isinstance(arr, np.ndarray):
            continue
        if np.random.rand() < 0.1:
            maps[nm] = cv2.GaussianBlur(arr, (5, 5), 0)
        if np.random.rand() < 0.1:
            maps[nm] = np.clip(arr * (0.5 + np.random.rand() * 0.65), 0, 255)
        if np.random.rand() < 0.1:
            gr = arr.mean(axis=2, keepdims=True)
            r = np.random.rand()
            maps[nm] = arr * r + gr * (1 - r)
    return maps


def _to_tensors(maps: dict) -> dict:
    """convert all numpy arrays to float32 tensors.
    hwc -> chw for 3-channel maps, hw stays hw."""
    out = {}
    for nm, val in maps.items():
        if not isinstance(val, np.ndarray):
            continue
        arr = val.astype(np.float32)
        if arr.ndim == 3:
            # hwc -> chw
            arr = np.transpose(arr, (2, 0, 1))
        out[nm] = torch.from_numpy(np.ascontiguousarray(arr))
    return out


# ---------------------------------------------------------------------------
# TransProteus dataset
# ---------------------------------------------------------------------------

class TransProteusDataset(Dataset):
    """
    pytorch dataset for the TransProteus synthetic vessel/content dataset.

    args:
        main_dir   : root directory of the dataset
        img_size   : (h, w) output spatial size; pass None to keep original size
        augment    : whether to apply colour augmentation during training
    """

    def __init__(self, main_dir: str, img_size=(512, 512), augment: bool = True):
        self.img_size = img_size
        self.augment = augment
        self.ann_list = []

        print(f"scanning {main_dir} ...")
        for ann_dir_name in os.listdir(main_dir):
            if ann_dir_name == ".DS_Store":
                continue
            ann_dir = os.path.join(main_dir, ann_dir_name)
            if not os.path.isdir(ann_dir):
                continue

            base = {}
            for json_key, fname in [
                ("ContentMaterial", "ContentMaterial.json"),
                ("VesselMaterial", "VesselMaterial.json"),
                ("CameraParameters", "CameraParameters.json"),
            ]:
                fp = os.path.join(ann_dir, fname)
                if os.path.isfile(fp):
                    base[json_key] = fp

            base["VesselMask"] = os.path.join(ann_dir, "VesselMask.png")
            base["VesselOpening_Depth"] = os.path.join(ann_dir, "VesselOpening_Depth.exr")
            base["EmptyVessel_RGB"] = os.path.join(ann_dir, "EmptyVessel_Frame_0_RGB.jpg")
            base["EmptyVessel_Normal"] = os.path.join(ann_dir, "EmptyVessel_Frame_0_Normal.exr")
            base["EmptyVessel_Depth"] = os.path.join(ann_dir, "EmptyVessel_Frame_0_Depth.exr")
            base["MainDir"] = ann_dir

            for nm in os.listdir(ann_dir):
                if "VesselWithContent" not in nm or "_RGB.jpg" not in nm:
                    continue
                fp = os.path.join(ann_dir, nm)
                entry = base.copy()
                entry["VesselWithContentRGB"] = fp
                entry["VesselWithContentNormal"] = fp.replace("_RGB.jpg", "_Normal.exr")
                entry["VesselWithContentDepth"] = fp.replace("_RGB.jpg", "_Depth.exr")
                entry["ContentRGB"] = fp.replace("VesselWithContent_", "Content_")
                entry["ContentNormal"] = entry["VesselWithContentNormal"].replace(
                    "VesselWithContent_", "Content_"
                )
                entry["ContentDepth"] = entry["VesselWithContentDepth"].replace(
                    "VesselWithContent_", "Content_"
                )
                entry["ContentMask"] = entry["ContentDepth"].replace("_Depth.exr", "_Mask.png")
                self.ann_list.append(entry)

        # validate paths up front so errors surface early with a clear message
        missing = []
        for entry in self.ann_list:
            for key, path in entry.items():
                if any(path.endswith(ext) for ext in (".exr", ".png", ".jpg")):
                    if not os.path.exists(path):
                        missing.append(path)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} file(s) missing from dataset. first: {missing[0]}"
            )

        print(f"TransProteus: {len(self.ann_list)} samples ready.")

    def __len__(self):
        return len(self.ann_list)

    def __getitem__(self, idx):
        ann = self.ann_list[idx]
        maps = {}

        # load all image/depth/mask files
        for nm, depth in MAPS_AND_DEPTHS.items():
            if nm not in ann:
                continue
            path = ann[nm]
            if ".exr" in path:
                img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                if img is None:
                    raise IOError(f"failed to read {path}")
                if img.ndim >= 3 and depth == 1:
                    img = img[:, :, 0]
            else:
                img = cv2.imread(path, 0 if depth == 1 else cv2.IMREAD_COLOR)
                if img is None:
                    raise IOError(f"failed to read {path}")
            maps[nm] = img.astype(np.float32)

        # derived masks
        maps["VesselMask"] = (maps["VesselMask"] > 0).astype(np.float32)
        maps["VesselOpeningMask"] = (maps["VesselOpening_Depth"] < 5000).astype(np.float32)
        maps["ContentMaskClean"] = (maps["ContentMask"].sum(2) > 0).astype(np.float32)
        maps["ROI"] = np.ones(maps["VesselMask"].shape, dtype=np.float32)

        ignore = maps["ContentMask"][:, :, 2].copy()
        ignore[maps["ContentMask"][:, :, 1] > 0] = 0
        ignore[(maps["ContentMask"][:, :, 1] * maps["ContentMask"][:, :, 0]) > 0] = 1
        maps["ROI"][ignore > 0] = 0

        # clamp far-away depth values
        for key in ("EmptyVessel_Depth", "VesselOpening_Depth", "ContentDepth"):
            if key in maps:
                maps[key][maps[key] > 5000] = 0

        if self.augment:
            maps = _augment(maps)
        if self.img_size is not None:
            maps = _crop_resize(maps, self.img_size[0], self.img_size[1])

        return _to_tensors(maps)


# ---------------------------------------------------------------------------
# LabPics dataset
# ---------------------------------------------------------------------------

class LabPicsDataset(Dataset):
    """
    pytorch dataset for the LabPics real-world lab imagery dataset.

    args:
        main_dir   : root directory of the dataset
        img_size   : (h, w) output spatial size; pass None to keep original size
        augment    : whether to apply colour/flip augmentation during training
    """

    def __init__(self, main_dir: str, img_size=(512, 512), augment: bool = True):
        self.img_size = img_size
        self.augment = augment

        print(f"scanning {main_dir} ...")
        self.ann_list = [
            os.path.join(main_dir, d)
            for d in os.listdir(main_dir)
            if d != ".DS_Store"
        ]
        print(f"LabPics: {len(self.ann_list)} samples ready.")

    def __len__(self):
        return len(self.ann_list)

    def __getitem__(self, idx):
        path = self.ann_list[idx]
        sem_dir = os.path.join(path, "SemanticMaps", "FullImage")

        img = cv2.imread(os.path.join(path, "Image.jpg"))
        if img is None:
            raise IOError(f"failed to read image at {path}/Image.jpg")
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=2)
        img = img[:, :, :3]

        def _load_mask(fpath):
            m = cv2.imread(fpath)
            return m if m is not None else np.zeros(img.shape, dtype=np.uint8)

        vessel_mask = _load_mask(os.path.join(sem_dir, "Transparent.png"))
        filled_mask = _load_mask(os.path.join(sem_dir, "Filled.png"))
        parts_mask = _load_mask(os.path.join(sem_dir, "PartInsideVessel.png"))
        mat_scattered = _load_mask(os.path.join(sem_dir, "MaterialScattered.png"))

        ignore_path = os.path.join(path, "Ignore.png")
        ignore = cv2.imread(ignore_path, 0) if os.path.exists(ignore_path) else np.zeros(img.shape[:2], dtype=np.uint8)

        maps = {}
        maps["VesselWithContentRGB"] = img.astype(np.float32)
        maps["VesselMask"] = ((vessel_mask[:, :, 0] > 0) | (parts_mask[:, :, 0] > 0)).astype(np.float32)
        maps["ROI"] = (1 - (ignore > 0).astype(np.float32))
        maps["ROI"][filled_mask[:, :, 2] > 15] = 0
        maps["ROI"][mat_scattered[:, :, 2] > 0] = 0
        maps["ContentMaskClean"] = (filled_mask[:, :, 0] > 0).astype(np.float32) * maps["VesselMask"]

        if self.augment:
            if np.random.rand() < 0.5:  # horizontal flip
                for nm in maps:
                    if isinstance(maps[nm], np.ndarray):
                        maps[nm] = np.fliplr(maps[nm])
            maps = _augment(maps)

        if self.img_size is not None:
            maps = _crop_resize(maps, self.img_size[0], self.img_size[1])

        return _to_tensors(maps)


# ---------------------------------------------------------------------------
# collate helper — handles dicts of tensors with potentially different keys
# ---------------------------------------------------------------------------

def collate_maps(batch: list) -> dict:
    """stack a list of map-dicts into a single batched map-dict.
    keys absent in some samples are skipped."""
    keys = batch[0].keys()
    return {k: torch.stack([s[k] for s in batch]) for k in keys if all(k in s for s in batch)}


# ---------------------------------------------------------------------------
# convenience factory
# ---------------------------------------------------------------------------

def make_dataloader(
    datasets,
    batch_size: int = 4,
    num_workers: int = 8,
    shuffle: bool = True,
) -> DataLoader:
    """
    build a dataloader from one or more dataset instances.

    args:
        datasets    : a single Dataset or a list of Dataset objects to combine
        batch_size  : samples per batch
        num_workers : parallel data loading workers (rule of thumb: 4-8 per gpu)
        shuffle     : shuffle between epochs

    example:
        train_loader = make_dataloader([
            TransProteusDataset("/data/TransProteus", img_size=(512, 512)),
            LabPicsDataset("/data/LabPics", img_size=(512, 512)),
        ])
        for batch in train_loader:
            rgb = batch["VesselWithContentRGB"].cuda()  # (B, 3, H, W)
    """
    if isinstance(datasets, (list, tuple)):
        dataset = ConcatDataset(datasets)
    else:
        dataset = datasets

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,        # faster cpu->gpu transfer
        persistent_workers=True if num_workers > 0 else False,  # avoid worker restart cost
        prefetch_factor=2 if num_workers > 0 else None,         # prefetch next batch while gpu trains
        collate_fn=collate_maps,
        drop_last=True,         # keeps batch size consistent; avoids batchnorm issues
    )


# ---------------------------------------------------------------------------
# usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_loader = make_dataloader(
        datasets=[
            TransProteusDataset("/data/TransProteus", img_size=(512, 512), augment=True),
            LabPicsDataset("/data/LabPics", img_size=(512, 512), augment=True),
        ],
        batch_size=4,
        num_workers=8,
    )

    for batch in train_loader:
        rgb = batch["VesselWithContentRGB"]   # (B, 3, H, W) float32 tensor
        mask = batch["VesselMask"]            # (B, H, W) float32 tensor
        print(rgb.shape, mask.shape)
        break
