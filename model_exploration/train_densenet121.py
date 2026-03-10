import os
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pydicom
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight


# =========================
# CONFIG
# =========================
LABELS_CSV = "/Users/tanmayswami/Downloads/stage_2_train_labels.csv"
IMAGES_DIR = "/Users/tanmayswami/Downloads/stage_2_train_images"

SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 2        
LR = 1e-4
EPOCHS = 10
WEIGHT_DECAY = 1e-4
THRESH = 0.5
IMG_SIZE = 224
PATIENCE = 3
MIN_DELTA = 1e-4

BEST_MODEL_PATH = "rsna_densenet121_best_f1.pt"
LAST_MODEL_PATH = "rsna_densenet121_last.pt"


# =========================
# REPRODUCIBILITY
# =========================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================
# DEVICE
# =========================
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =========================
# DICOM HELPERS
# =========================
def dicom_to_pil(path: str) -> Image.Image:
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img

    img -= img.min()
    if img.max() > 0:
        img /= img.max()

    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img).convert("RGB")


def index_images(images_dir: str) -> pd.DataFrame:
    rows = []
    p = Path(images_dir)

    for fp in p.glob("*.dcm"):
        rows.append({"patientId": fp.stem, "path": str(fp)})

    out = pd.DataFrame(rows).drop_duplicates(subset=["patientId"])
    if out.empty:
        raise ValueError(f"No .dcm files found in: {images_dir}")
    return out


def build_patient_labels(labels_csv: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)

    if "Target" not in df.columns or "patientId" not in df.columns:
        raise ValueError("Expected columns: patientId, Target")

    y = df.groupby("patientId")["Target"].max().reset_index()
    y.rename(columns={"Target": "y"}, inplace=True)
    return y


def build_rsna_df(labels_csv: str, images_dir: str) -> pd.DataFrame:
    y_df = build_patient_labels(labels_csv)
    img_df = index_images(images_dir)

    df = img_df.merge(y_df, on="patientId", how="inner")
    if df.empty:
        raise ValueError("No overlap between labels and images.")

    return df


# =========================
# DATASET
# =========================
class RSNADatasetTorchvision(Dataset):
    def __init__(self, df: pd.DataFrame, tfm: transforms.Compose):
        self.df = df.reset_index(drop=True)
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = dicom_to_pil(row["path"])
        y = int(row["y"])
        x = self.tfm(img)
        return x, y


# =========================
# MODEL
# =========================
def build_densenet121(num_classes: int = 2) -> nn.Module:
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


# =========================
# TRAIN / EVAL UTILS
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        n += bs

    return total_loss / max(n, 1)


@torch.no_grad()
def infer_probs(model, loader, device):
    model.eval()
    all_probs = []
    all_y = []
    t0 = time.time()

    for x, y in tqdm(loader, desc="Infer", leave=False):
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        all_probs.append(probs)
        all_y.append(y.numpy())

    t1 = time.time()

    probs = np.vstack(all_probs)
    y_true = np.concatenate(all_y)
    elapsed = t1 - t0
    ips = len(y_true) / elapsed if elapsed > 0 else float("inf")

    return probs, y_true, elapsed, ips


def evaluate_binary_from_probs(probs, y_true, pos_idx=1, thresh=0.5):
    y_score = probs[:, pos_idx]
    y_pred = (y_score >= thresh).astype(int)

    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    acc = accuracy_score(y_true, y_pred)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    return {
        "auc": auc,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "cm": cm,
        "y_score": y_score,
        "y_pred": y_pred,
    }


def save_checkpoint(path, model, threshold, config, best_epoch=None, best_f1=None):
    payload = {
        "model_name": "densenet121",
        "state_dict": model.state_dict(),
        "threshold": threshold,
        "config": config,
    }

    if best_epoch is not None:
        payload["best_epoch"] = best_epoch
    if best_f1 is not None:
        payload["best_f1"] = best_f1

    torch.save(payload, path)


def load_densenet_checkpoint(ckpt_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_densenet121(num_classes=2).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# =========================
# MAIN
# =========================
def main():
    set_seed(SEED)
    device = get_device()

    print("=" * 60)
    print("Device:", device)
    print("MPS available:", torch.backends.mps.is_available())
    print("NUM_WORKERS:", NUM_WORKERS)
    print("=" * 60)

    # -------------------------
    # Build full dataframe
    # -------------------------
    df_all = build_rsna_df(LABELS_CSV, IMAGES_DIR)

    print("Total matched:", len(df_all))
    print(df_all["y"].value_counts())
    print("Pos rate:", df_all["y"].mean())

    # -------------------------
    # 70 / 15 / 15 stratified split
    # -------------------------
    train_df, temp_df = train_test_split(
        df_all,
        test_size=0.30,
        stratify=df_all["y"],
        random_state=SEED
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["y"],
        random_state=SEED
    )

    print("\nSplit sizes:")
    print("Train:", len(train_df), "| Pos rate:", train_df["y"].mean())
    print("Val  :", len(val_df),   "| Pos rate:", val_df["y"].mean())
    print("Test :", len(test_df),  "| Pos rate:", test_df["y"].mean())

    # -------------------------
    # Transforms
    # -------------------------
    train_tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    eval_tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    # -------------------------
    # Datasets
    # -------------------------
    train_ds = RSNADatasetTorchvision(train_df, train_tfm)
    val_ds = RSNADatasetTorchvision(val_df, eval_tfm)
    test_ds = RSNADatasetTorchvision(test_df, eval_tfm)

    # On macOS/MPS, keep this simple first
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    # -------------------------
    # Model
    # -------------------------
    model = build_densenet121(num_classes=2).to(device)

    # class weights from TRAIN only
    classes = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["y"].values
    )
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print("Class weights:", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    config = {
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "num_workers": NUM_WORKERS,
        "img_size": IMG_SIZE,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
    }

    # -------------------------
    # Training loop
    # -------------------------
    best_f1 = -1.0
    best_epoch = -1
    best_metrics = None
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_probs, val_y, val_elapsed, val_ips = infer_probs(model, val_loader, device)
        val_metrics = evaluate_binary_from_probs(
            val_probs,
            val_y,
            pos_idx=1,
            thresh=THRESH
        )

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Val AUROC: {val_metrics['auc']:.4f} | Acc: {val_metrics['acc']:.4f}")
        print(
            f"Precision: {val_metrics['prec']:.4f} | "
            f"Recall: {val_metrics['rec']:.4f} | "
            f"F1: {val_metrics['f1']:.4f}"
        )
        print("Val Confusion Matrix [[TN FP],[FN TP]]:")
        print(val_metrics["cm"])
        print(
            f"Val elapsed: {val_elapsed:.2f}s | "
            f"Throughput: {val_ips:.2f} imgs/s | "
            f"Batch: {BATCH_SIZE} | "
            f"Thresh: {THRESH}"
        )

        improved = val_metrics["f1"] > (best_f1 + MIN_DELTA)

        if improved:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            best_metrics = val_metrics
            epochs_without_improvement = 0

            save_checkpoint(
                BEST_MODEL_PATH,
                model,
                threshold=THRESH,
                config=config,
                best_epoch=best_epoch,
                best_f1=best_f1
            )
            print(f"Saved BEST model to: {BEST_MODEL_PATH}")
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement in val F1 for {epochs_without_improvement} epoch(s). "
                f"Patience: {PATIENCE}"
            )

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break



    #     if val_metrics["f1"] > best_f1:
    #         best_f1 = val_metrics["f1"]
    #         best_epoch = epoch
    #         best_metrics = val_metrics

    #         save_checkpoint(
    #             BEST_MODEL_PATH,
    #             model,
    #             threshold=THRESH,
    #             config=config,
    #             best_epoch=best_epoch,
    #             best_f1=best_f1
    #         )
    #         print(f"Saved BEST model to: {BEST_MODEL_PATH}")

    # # save last model
    save_checkpoint(
        LAST_MODEL_PATH,
        model,
        threshold=THRESH,
        config=config
    )
    print(f"\nSaved LAST model to: {LAST_MODEL_PATH}")
    print(f"Best epoch by val F1: {best_epoch} | Best val F1: {best_f1:.4f}")

    if best_metrics is not None:
        print("Best validation confusion matrix:")
        print(best_metrics["cm"])

    # -------------------------
    # Final test using best model
    # -------------------------
    best_model = load_densenet_checkpoint(BEST_MODEL_PATH, device)

    test_probs, test_y, test_elapsed, test_ips = infer_probs(best_model, test_loader, device)
    test_metrics = evaluate_binary_from_probs(
        test_probs,
        test_y,
        pos_idx=1,
        thresh=THRESH
    )

    print("\n===== FINAL TEST RESULTS =====")
    print(f"Test AUROC: {test_metrics['auc']:.4f} | Acc: {test_metrics['acc']:.4f}")
    print(
        f"Precision: {test_metrics['prec']:.4f} | "
        f"Recall: {test_metrics['rec']:.4f} | "
        f"F1: {test_metrics['f1']:.4f}"
    )
    print("Test Confusion Matrix [[TN FP],[FN TP]]:")
    print(test_metrics["cm"])
    print(f"Test elapsed: {test_elapsed:.2f}s | Throughput: {test_ips:.2f} imgs/s")

    print("\nClassification report (TEST):")
    print(classification_report(
        test_y,
        test_metrics["y_pred"],
        target_names=["NORMAL", "PNEUMONIA"],
        zero_division=0
    ))


if __name__ == "__main__":
    main()