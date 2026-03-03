import os
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import src.models.model as NET_FCN
import src.models.loss_functions as LossFunctions
import src.data.make_dataset as MakeDataset

"""
training loop for the segmentation and depth prediction model.

the main function takes:
    --batch_size            : samples per batch
    --num_epochs            : number of training epochs
    --load_pretrained_model : whether to resume from a saved checkpoint
    --use_labpics           : whether to include labpics data in training
"""

DEPTH_LIST = ["EmptyVessel_Depth", "ContentDepth", "VesselOpening_Depth"]
MASK_LIST = ["VesselMask", "ContentMaskClean", "VesselOpeningMask"]
DEPTH_TO_MASK = {
    "EmptyVessel_Depth": "VesselMask",
    "ContentDepth": "ContentMaskClean",
    "VesselOpening_Depth": "VesselOpeningMask",
}


def train(batch_size, num_epochs, load_pretrained_model, use_labpics):
    learning_rate = 1e-5
    weight_decay = 4e-5
    trained_model_weight_dir = "logs/"
    train_loss_txt = (
        trained_model_weight_dir
        + "TrainLoss_"
        + time.strftime("%d%m%Y-%H%M")
        + ".txt"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- model setup ---
    net = NET_FCN.Net(MaskList=MASK_LIST, DepthList=DEPTH_LIST)

    init_epoch = 1
    if load_pretrained_model:
        ckpt = "models/40__29032023-0231.torch"
        if os.path.exists(ckpt):
            print(f"loading pretrained model from {ckpt}")
            net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        lr_path = os.path.join(trained_model_weight_dir, "Learning_Rate.npy")
        ep_path = os.path.join(trained_model_weight_dir, "epoch.npy")
        if os.path.exists(lr_path):
            learning_rate = float(np.load(lr_path))
        if os.path.exists(ep_path):
            init_epoch = int(np.load(ep_path))

    net = net.to(device)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    os.makedirs(trained_model_weight_dir, exist_ok=True)
    # sanity-check that saving works before training starts
    torch.save(
        net.state_dict(), os.path.join(trained_model_weight_dir, "test.torch")
    )

    # --- dataloader ---
    # the loader yields batches that are already float32 tensors shaped (B, C, H, W)
    # (or (B, H, W) for single-channel maps). no manual numpy->tensor conversion needed.
    train_loader, num_samples = MakeDataset.create_train_loader(
        batch_size=batch_size,
        use_labpics=use_labpics,
    )
    itr_per_epoch = len(train_loader)
    print(
        f"total samples: {num_samples}, iterations per epoch: {itr_per_epoch}"
    )

    avg_cat_loss = {}
    init_step = 1

    print("starting training")
    for epoch_num in range(init_epoch, num_epochs):
        print(f"epoch {epoch_num}")

        # each batch is already a dict of tensors thanks to collate_maps
        for itr, gt in enumerate(tqdm(train_loader), start=1):

            # move everything to device once here — tensors are pinned so this is async
            gt = {k: v.to(device, non_blocking=True) for k, v in gt.items()}

            # --- forward pass ---
            # the loader outputs (B, 3, H, W) for RGB maps,
            # so no manual unsqueeze or numpy conversion is needed
            prd_depth, prd_prob, prd_mask = net(
                Images=gt["VesselWithContentRGB"]
            )

            net.zero_grad()

            cat_loss = {}

            # depth loss — only available for transproteus batches.
            # labpics batches won't have depth keys in gt, so we skip gracefully.
            for nm in DEPTH_LIST:
                if nm not in gt:
                    continue

                roi = (gt[DEPTH_TO_MASK[nm]] * gt["ROI"]).unsqueeze(
                    1
                )  # (B,1,H,W)
                roi = nn.functional.interpolate(
                    roi,
                    size=(prd_depth[nm].shape[2], prd_depth[nm].shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
                roi = (roi > 0.9).float()

                tgt_depth = torch.log(gt[nm].unsqueeze(1) + 0.0001)
                tgt_depth = nn.functional.interpolate(
                    tgt_depth,
                    size=(prd_depth[nm].shape[2], prd_depth[nm].shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )

                cat_loss[nm] = 5 * LossFunctions.DepthLoss(
                    prd_depth[nm], tgt_depth, roi
                )

            # segmentation mask loss
            roi = gt["ROI"].unsqueeze(1)  # (B,1,H,W)
            roi = nn.functional.interpolate(
                roi,
                size=(
                    prd_prob[MASK_LIST[0]].shape[2],
                    prd_prob[MASK_LIST[0]].shape[3],
                ),
                mode="bilinear",
                align_corners=False,
            )

            for nm in MASK_LIST:
                if nm not in gt:
                    continue
                tgt = gt[nm].unsqueeze(1).float()  # (B,1,H,W)
                tgt = nn.functional.interpolate(
                    tgt,
                    size=(prd_prob[nm].shape[2], prd_prob[nm].shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
                cat_loss[nm] = -torch.mean(
                    tgt[:, 0]
                    * torch.log(prd_prob[nm][:, 1] + 1e-5)
                    * roi[:, 0]
                ) - torch.mean(
                    (1 - tgt[:, 0])
                    * torch.log(prd_prob[nm][:, 0] + 1e-7)
                    * roi[:, 0]
                )

            # --- loss aggregation and backprop ---
            global_itr = (epoch_num - 1) * itr_per_epoch + itr - init_step + 1
            fr = 1 / min(global_itr, 2000)

            total_loss = sum(cat_loss.values())

            for k in ("Depth", "Mask", "Total"):
                avg_cat_loss.setdefault(k, 0)

            for nm, loss_val in cat_loss.items():
                avg_cat_loss.setdefault(nm, 0)
                if loss_val > 0:
                    scalar = loss_val.detach().cpu().item()
                    avg_cat_loss[nm] = (1 - fr) * avg_cat_loss[
                        nm
                    ] + fr * scalar
                if "Depth" in nm:
                    avg_cat_loss["Depth"] += avg_cat_loss[nm]
                if "Mask" in nm:
                    avg_cat_loss["Mask"] += avg_cat_loss[nm]
                avg_cat_loss["Total"] += avg_cat_loss[nm]

            total_loss.backward()
            optimizer.step()

            # log every 4 iterations
            if itr % 4 == 0:
                txt = "\n" + str(global_itr)
                for nm, v in avg_cat_loss.items():
                    txt += f"\taverage cat loss [{nm}] {float(v):.4f}"
                with open(train_loss_txt, "a") as f:
                    f.write(txt)

        # --- end of epoch ---
        tqdm.write(f"epoch {epoch_num} complete")

        torch.save(
            net.state_dict(),
            os.path.join(trained_model_weight_dir, "Default.torch"),
        )
        torch.save(
            net.state_dict(),
            os.path.join(trained_model_weight_dir, "DefaultBackUp.torch"),
        )
        np.save(
            os.path.join(trained_model_weight_dir, "Learning_Rate.npy"),
            learning_rate,
        )
        np.save(os.path.join(trained_model_weight_dir, "epoch.npy"), epoch_num)

        if epoch_num % 5 == 0:
            save_path = (
                f"models/{epoch_num}__{time.strftime('%d%m%Y-%H%M')}.torch"
            )
            torch.save(net.state_dict(), save_path)
            print(f"model saved to {save_path}")

        # learning rate schedule
        if epoch_num % 5 == 0:
            prev = avg_cat_loss.get("TotalPrevious")
            if prev is None or avg_cat_loss["Total"] * 0.95 < prev:
                learning_rate *= 0.9
                if learning_rate <= 3e-7:
                    learning_rate = 5e-6
                print(f"learning rate updated to {learning_rate}")
                optimizer = torch.optim.Adam(
                    net.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            avg_cat_loss["TotalPrevious"] = avg_cat_loss["Total"] + 1e-10
