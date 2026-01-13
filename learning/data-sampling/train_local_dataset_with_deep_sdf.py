import torch
from torch.utils.data import Dataset, DataLoader, random_split
import signal
import sys
import os
import logging
import math
import json
import time
from pathlib import Path
import numpy as np
from typing import Tuple

# 📌 ASSUMPTION: The Decoder class from deep_sdf_decoder.py is now importable.
try:
    from deep_sdf_decoder import Decoder
except ImportError:
    print("FATAL ERROR: Could not import Decoder from deep_sdf_decoder.py.")
    print("Please ensure deep_sdf_decoder.py is in a location accessible by Python.")
    sys.exit(1)

# --- 1. Data Utilities (reused) ---


class SDFDataset(Dataset):
    """Loads SDF data from a single .npy file: [x, y, z, d]."""

    def __init__(self, data_file_path: Path):
        if not data_file_path.exists():
            raise FileNotFoundError(f"Data file not found at: {data_file_path}")

        print(f"Loading data from: {data_file_path.name}...")
        self.data = np.load(data_file_path)

        self.points = torch.tensor(self.data[:, :3], dtype=torch.float32)
        self.sdfs = torch.tensor(self.data[:, 3:], dtype=torch.float32)

        print(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        # We also return a scene index (0) since we only have one scene/link
        return self.points[idx], self.sdfs[idx], torch.zeros(1, dtype=torch.long)


def create_dataloaders(
    data_file_path: Path,
    batch_size: int = 4096,
    val_split_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Creates train and validation DataLoaders for a given SDF data file."""
    torch.manual_seed(seed)
    full_dataset = SDFDataset(data_file_path)
    total_size = len(full_dataset)
    val_size = int(val_split_ratio * total_size)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    num_workers = min(os.cpu_count(), 8)

    # Collate function is needed to handle the index batching correctly
    def custom_collate(batch):
        # batch is a list of (points, sdfs, index) tuples
        points = torch.vstack([item[0] for item in batch])
        sdfs = torch.vstack([item[1] for item in batch])
        # Indices are all 0, but we need N x 1 shape
        indices = torch.zeros((len(batch), 1), dtype=torch.long)
        return points, sdfs, indices.squeeze(1)  # Indices shape: (N,)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate,
    )

    print(
        f"\nData Loaders created: Train samples: {train_size}, Val samples: {val_size}"
    )
    return train_dataloader, val_dataloader


# --- 2. Supporting/Helper Functions (Minimal required) ---
# (Learning rate schedules, save functions, etc. are assumed to be implemented or imported)


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


def get_learning_rate_schedules(specs):
    # This is simplified; in a full setup, it would parse all schedules
    schedule_specs = specs["LearningRateSchedule"]
    schedules = []
    # Schedule 0 for Decoder, Schedule 1 for Latent Code
    schedules.append(ConstantLearningRateSchedule(schedule_specs[0]["Value"]))
    schedules.append(ConstantLearningRateSchedule(schedule_specs[1]["Value"]))
    return schedules


def adjust_learning_rate(lr_schedules, optimizer_decoder, optimizer_latent, epoch):
    """Updates the learning rate for both optimizers."""
    # Decoder optimizer is always index 0
    optimizer_decoder.param_groups[0]["lr"] = lr_schedules[0].get_learning_rate(epoch)
    # Latent optimizer is always index 1
    optimizer_latent.param_groups[0]["lr"] = lr_schedules[1].get_learning_rate(epoch)


def save_model(experiment_directory, filename, decoder, epoch):
    model_params_dir = Path(experiment_directory) / "ModelParams"
    model_params_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch, name="Optimizer"):
    optimizer_params_dir = Path(experiment_directory) / f"{name}Params"
    optimizer_params_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def save_latent_codes(experiment_directory, filename, lat_vecs, epoch):
    latent_codes_dir = Path(experiment_directory) / "LatentCodes"
    latent_codes_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"epoch": epoch, "latent_codes": lat_vecs.state_dict()},
        os.path.join(latent_codes_dir, filename),
    )


def get_latent_size_loss(latent_vecs):
    """L2 regularization loss on the latent vectors."""
    # This is the sum of the L2 norms of all latent vectors
    return torch.mean(torch.sum(latent_vecs**2, dim=1))


# --- 3. Main Training Function (ADAPTED) ---


def train_sdf_model(
    experiment_directory_str: str, data_file_path_str: str, batch_split: int = 1
):

    experiment_directory = Path(experiment_directory_str)
    data_file_path = Path(data_file_path_str)
    experiment_directory.mkdir(parents=True, exist_ok=True)

    # --- Configuration (Simulating specs.json) ---
    NUM_SCENES = 1  # We have only one mesh/link
    LATENT_SIZE = 256  # Standard DeepSDF latent size
    CODE_BOUND = 1.0  # Max norm for the latent vector
    LATENT_REGULARIZATION_WEIGHT = 1e-4  # Lambda in the loss function

    specs = {
        "Description": "Single-Link DeepSDF Model Training",
        "CodeLength": LATENT_SIZE,
        "NetworkSpecs": {
            "dims": [512, 512, 512, 512, 512, 512, 512, 512],
            "latent_in": [4],
            "weight_norm": True,
            "latent_dropout": True,
            "xyz_in_all": False,
        },
        "NumEpochs": 5000,
        "SnapshotFrequency": 100,
        # Two schedules: [0] for Decoder, [1] for Latent Code (typically lower LR)
        "LearningRateSchedule": [
            {"Type": "Constant", "Value": 1e-4},
            {"Type": "Constant", "Value": 1e-5},
        ],
        "LogFrequency": 10,
        "GradientClipNorm": None,
    }

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info(f"Starting training in {experiment_directory}")

    # --- Model Setup ---
    latent_size = specs["CodeLength"]
    decoder = Decoder(latent_size, **specs["NetworkSpecs"]).cuda()

    # Latent Vector Setup (Only one scene/link)
    lat_vecs = torch.nn.Embedding(NUM_SCENES, latent_size, max_norm=CODE_BOUND)
    # Initialize the latent vector
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        1.0 / math.sqrt(latent_size),
    )
    lat_vecs = lat_vecs.cuda()

    if torch.cuda.device_count() > 1:
        decoder = torch.nn.DataParallel(decoder)

    # --- Data Loading ---
    BATCH_SIZE = 4096
    train_loader, _ = create_dataloaders(
        data_file_path=data_file_path, batch_size=BATCH_SIZE
    )

    # --- Loss and Clamping Setup ---
    clamp_dist = 0.1
    minT = -clamp_dist
    maxT = clamp_dist
    loss_l1 = torch.nn.L1Loss(reduction="sum")

    # --- Optimizer Setup (Two separate optimizers) ---
    lr_schedules = get_learning_rate_schedules(specs)

    # Optimizer for Decoder weights
    optimizer_decoder = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            }
        ]
    )

    # Optimizer for Latent Vector
    optimizer_latent = torch.optim.Adam(
        [
            {
                "params": lat_vecs.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            }
        ]
    )

    grad_clip = specs.get("GradientClipNorm")

    # --- Training Loop ---
    for epoch in range(1, specs["NumEpochs"] + 1):

        start = time.time()
        decoder.train()

        # Update learning rates
        adjust_learning_rate(lr_schedules, optimizer_decoder, optimizer_latent, epoch)

        epoch_total_loss = 0.0

        # DataLoader now yields: xyz (points), sdf_gt (labels), indices (always 0)
        for xyz, sdf_gt, indices in train_loader:

            # 1. Move data to GPU
            # print(xyz)
            # print(sdf_gt)
            xyz = xyz.cuda()
            sdf_gt = sdf_gt.cuda()
            indices = indices.cuda()

            num_sdf_samples = xyz.shape[0]

            # 2. Clamp Ground Truth SDFs
            sdf_gt = torch.clamp(sdf_gt, minT, maxT)

            # 3. Fetch Latent Vector
            # indices is an (N, 1) tensor of all zeros
            batch_vecs = lat_vecs(indices)

            # 4. Concatenate Latent Vector and Points (N, L+3)
            input_data = torch.cat([batch_vecs.squeeze(1), xyz], dim=1)

            # Split the batch
            input_chunks = torch.chunk(input_data, batch_split)
            sdf_gt_chunks = torch.chunk(sdf_gt, batch_split)
            batch_vecs_chunks = torch.chunk(
                batch_vecs, batch_split
            )  # Needed for reg. loss

            # Reset gradients for both optimizers
            optimizer_decoder.zero_grad()
            optimizer_latent.zero_grad()

            batch_loss = 0.0

            for i in range(batch_split):

                # 5. Forward Pass
                pred_sdf = decoder(input_chunks[i])

                # 6. Clamp Prediction
                pred_sdf = torch.clamp(pred_sdf, minT, maxT)

                # 7. Calculate Losses
                # L1 Reconstruction Loss
                l1_loss = loss_l1(pred_sdf, sdf_gt_chunks[i]) / num_sdf_samples

                # Latent Regularization Loss
                l2_size_loss = get_latent_size_loss(batch_vecs_chunks[i])
                reg_loss = LATENT_REGULARIZATION_WEIGHT * l2_size_loss

                # Total Loss
                total_loss = l1_loss + reg_loss

                # 8. Backward Pass
                total_loss.backward()
                batch_loss += total_loss.item()

            # 9. Optimization Step
            if grad_clip is not None:
                # Apply gradient clipping to the decoder only
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

            optimizer_decoder.step()
            optimizer_latent.step()  # Latent vector is also updated

            epoch_total_loss += batch_loss * num_sdf_samples

        # Logging and Checkpointing
        avg_epoch_loss = epoch_total_loss / len(train_loader.dataset)
        seconds_elapsed = time.time() - start

        logging.info(
            "Epoch {} | Loss: {:.6f} | L1 Loss: {:.6f} | Reg Loss: {:.6f} | Time: {:.2f}s".format(
                epoch,
                avg_epoch_loss,
                l1_loss.item(),  # Use the last computed L1 loss chunk value for logging
                reg_loss.item(),  # Use the last computed Reg loss chunk value for logging
                seconds_elapsed,
            )
        )

        if epoch % specs["SnapshotFrequency"] == 0:
            save_model(experiment_directory, str(epoch) + ".pth", decoder, epoch)
            save_latent_codes(
                experiment_directory, str(epoch) + ".pth", lat_vecs, epoch
            )
            save_optimizer(
                experiment_directory,
                str(epoch) + ".pth",
                optimizer_decoder,
                epoch,
                name="DecoderOptimizer",
            )
            save_optimizer(
                experiment_directory,
                str(epoch) + ".pth",
                optimizer_latent,
                epoch,
                name="LatentOptimizer",
            )

        if epoch % specs["LogFrequency"] == 0:
            save_model(experiment_directory, "latest.pth", decoder, epoch)
            save_latent_codes(experiment_directory, "latest.pth", lat_vecs, epoch)
            save_optimizer(
                experiment_directory,
                "latest.pth",
                optimizer_decoder,
                epoch,
                name="DecoderOptimizer",
            )
            save_optimizer(
                experiment_directory,
                "latest.pth",
                optimizer_latent,
                epoch,
                name="LatentOptimizer",
            )
            # Save logs here


if __name__ == "__main__":

    # --- EXAMPLE USAGE TO TRAIN ONE LINK'S MODEL ---

    EXPERIMENT_DIRS = [
        "ur5e_base_link_model_latent",
        "ur5e_shoulder_link_model_latent",
        "ur5e_forearm_link_model_latent",
        "ur5e_upper_arm_link_model_latent",
        "ur5e_wrist1_link_model_latent",
        "ur5e_wrist2_link_model_latent",
        "ur5e_wrist3_link_model_latent",
    ]
    DATA_FILES = [
        "ur5e_local_sdf_datasets/base_sdf.npy",
        "ur5e_local_sdf_datasets/shoulder_link_sdf.npy",
        "ur5e_local_sdf_datasets/forearm_link_sdf.npy",
        "ur5e_local_sdf_datasets/upper_arm_link_sdf.npy",
        "ur5e_local_sdf_datasets/wrist_1_link_sdf.npy",
        "ur5e_local_sdf_datasets/wrist_2_link_sdf.npy",
        "ur5e_local_sdf_datasets/wrist_3_link_sdf.npy",
    ]

    # 📌 1. Set the directory where you want to save the trained model/logs
    # EXPERIMENT_DIR = "ur5e_base_link_model_latent"
    # EXPERIMENT_DIR = "ur5e_shoulder_link_model_latent"
    # EXPERIMENT_DIR = "ur5e_forearm_link_model_latent"
    # EXPERIMENT_DIR = "ur5e_upper_arm_link_model_latent"
    # EXPERIMENT_DIR = "ur5e_wrist1_link_model_latent"
    # EXPERIMENT_DIR = "ur5e_wrist2_link_model_latent"
    EXPERIMENT_DIR = "ur5e_wrist3_link_model_latent"
    # 📌 2. Set the path to the SDF data file for the specific link you want to train
    # DATA_FILE = "ur5e_local_sdf_datasets/base_sdf.npy"  # Use your actual data file path
    # DATA_FILE = "ur5e_local_sdf_datasets/shoulder_link_sdf.npy"  # Use your actual data file path
    # DATA_FILE = (
    #    "ur5e_local_sdf_datasets/forearm_link_sdf.npy"  # Use your actual data file path
    # )
    # DATA_FILE = "ur5e_local_sdf_datasets/upper_arm_link_sdf.npy"  # Use your actual data file path
    # DATA_FILE = (
    #   "ur5e_local_sdf_datasets/wrist_1_link_sdf.npy"  # Use your actual data file path
    # )
    # DATA_FILE = (
    #    "ur5e_local_sdf_datasets/wrist_2_link_sdf.npy"  # Use your actual data file path
    # )
    DATA_FILE = (
        "ur5e_local_sdf_datasets/wrist_3_link_sdf.npy"  # Use your actual data file path
    )
    for index in range(len(DATA_FILES)):
        if not Path(DATA_FILES[index]).exists():
            print(
                f"FATAL: Data file not found at {DATA_FILES[index]}. Please run the data generation script first."
            )
        else:
            print(
                f"Starting latent-based DeepSDF training for model: {EXPERIMENT_DIRS[index]} using data from: {DATA_FILES[index]}"
            )

            train_sdf_model(
                experiment_directory_str=EXPERIMENT_DIRS[index],
                data_file_path_str=DATA_FILES[index],
                batch_split=1,
            )
