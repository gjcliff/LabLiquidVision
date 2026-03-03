from torch.utils.data import ConcatDataset
from src.data.load_data import (
    TransProteusDataset,
    LabPicsDataset,
    make_dataloader,
)

"""
this file creates dataloaders for training and testing.
paths and size constraints are defined here.
"""

TRANS_PROTEUS_FOLDERS = [
    "data/interim/TranProteus1/Training/LiquidContent",
    "data/interim/TranProteus2/Training/LiquidContent",
    "data/interim/TranProteus3/Training/LiquidContent",
    "data/interim/TranProteus4/Training/LiquidContent",
    "data/interim/TranProteus5/Training/LiquidContent",
    "data/interim/TranProteus6/Training/LiquidContent",
    "data/interim/TranProteus7/Training/LiquidContent",
    "data/interim/TranProteus8/Training/LiquidContent",
]

LABPICS_FOLDER = "data/interim/LabPics Chemistry/Train"

IMG_SIZE = (512, 512)  # fixed spatial size passed to both datasets


def create_train_loader(
    batch_size: int, use_labpics: bool = True, num_workers: int = 8
):
    """
    builds a combined dataloader for all transproteus folders and optionally labpics.

    args:
        batch_size  : samples per batch
        use_labpics : whether to include labpics data
        num_workers : parallel loading workers

    returns:
        dataloader, total_num_samples
    """
    datasets = [
        TransProteusDataset(folder, img_size=IMG_SIZE, augment=True)
        for folder in TRANS_PROTEUS_FOLDERS
    ]

    if use_labpics:
        datasets.append(
            LabPicsDataset(LABPICS_FOLDER, img_size=IMG_SIZE, augment=True)
        )

    combined = ConcatDataset(datasets)
    loader = make_dataloader(
        combined, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    return loader, len(combined)


def create_test_loader(
    test_folder: str, batch_size: int, num_workers: int = 4
):
    """
    builds a dataloader for evaluation (no augmentation, no shuffle).

    args:
        test_folder : path to test data root
        batch_size  : samples per batch
        num_workers : parallel loading workers

    returns:
        dataloader, total_num_samples
    """
    dataset = TransProteusDataset(
        test_folder, img_size=IMG_SIZE, augment=False
    )
    loader = make_dataloader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    return loader, len(dataset)
