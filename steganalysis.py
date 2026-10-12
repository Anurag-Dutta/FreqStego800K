# %% [markdown]
# ## Libraries

# %%
import os
import glob
import json
import random
import time
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# %% [markdown]
# ## Configuration

# %%
RUN_TASK1 = True   # Stego Identification
RUN_TASK2 = True   # Algorithm Attribution

BASE = r"..."          # removed for blinded review
GLOBAL_SAVE_DIR = os.path.join(BASE, 'checkpoints', 'resnet18_experiments')
os.makedirs(GLOBAL_SAVE_DIR, exist_ok=True)

IMAGE_SIZE   = (224, 224)
BATCH_SIZE   = 32
EPOCHS       = 1       # smoke test, increase for full scale steganalysis
LR           = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 0       # keep 0 for stability on Windows
SEED         = 42


ALGOS = [
    'dct_lsb', 'dct_phase', 'dct_zigzag',
    'dft_lsb', 'dft_phase', 'dft_radial',
    'dst_lsb', 'dwt_bp4', 'dwt_cross', 'dwt_lsb'
]
ALGO_TO_ID = {alg: i + 1 for i, alg in enumerate(ALGOS)}  # 1..10, 0 reserved for cover
ID_TO_ALGO = {v: k for k, v in ALGO_TO_ID.items()}


DATASETS = {
    'AI_ArtBench'          : (os.path.join(BASE, 'AI_ArtBench',          'Cover'), 'painting'),
    # get the full dataset from https://ieee-dataport.org/documents/ai-artbench
    'animal_faces_hq_afhq' : (os.path.join(BASE, 'animal_faces_hq_afhq', 'Cover'), 'animal_face'),
    # get the full dataset from https://wenku.csdn.net/doc/1u6v2evmxe
    'BOSSbase_1.01'        : (os.path.join(BASE, 'BOSSbase_1.01',        'Cover'), 'natural_gray'),
    # get the full dataset from https://dde.binghamton.edu/download/
    # direct download link -- http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip
    'BOWS2'                : (os.path.join(BASE, 'BOWS2',                'Cover'), 'natural_gray'),
    # get the full dataset from https://data.mendeley.com/datasets/kb3ngxfmjw/1
    'emonet_faces'         : (os.path.join(BASE, 'EmoNet-Face',          'Cover'), 'faces'),
    # get the full dataset from https://huggingface.co/datasets/laion/emonet-face-big
    'happywhales'          : (os.path.join(BASE, 'happywhales',          'Cover'), 'wildlife'),
    # get the full dataset from https://www.kaggle.com/competitions/happy-whale-and-dolphin
    'landcover'            : (os.path.join(BASE, 'landcover',            'Cover'), 'aerial'),
    # get the full dataset from https://huggingface.co/datasets/dragon7/LandCover.ai
    'wikiart'              : (os.path.join(BASE, 'wikiart',              'Cover'), 'art'),
    # get the full dataset from https://huggingface.co/datasets/huggan/wikiart
}


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# %% [markdown]
# ## Model Definition

# %% [markdown]
# e.g., ResNet-18

# %%
class ResNet18Binary(nn.Module):
    """ResNet-18 for binary classification (Task 1)."""
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)
        self._init_head()

    def _init_head(self):
        nn.init.normal_(self.backbone.fc.weight, 0, 0.01)
        nn.init.constant_(self.backbone.fc.bias, 0.0)

    def forward(self, x):
        return self.backbone(x)  # [B, 1]


class ResNet18Multi(nn.Module):
    """ResNet-18 for multi-class classification (Task 2). num_classes=11."""
    def __init__(self, num_classes=11):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        self._init_head()

    def _init_head(self):
        nn.init.normal_(self.backbone.fc.weight, 0, 0.01)
        nn.init.constant_(self.backbone.fc.bias, 0.0)

    def forward(self, x):
        return self.backbone(x)  # [B, 11]


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")

# %% [markdown]
# ## Indexing and Dataset Prep.

# %%
def build_index_for_dataset(ds_name):
    """For Task 1: per-cover index with stego dict."""
    cover_dir, domain = DATASETS[ds_name]
    EXT = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.pgm']

    cover_records = []
    stego_index = defaultdict(dict)

    cover_paths = []
    for ext in EXT:
        cover_paths.extend(glob.glob(os.path.join(cover_dir, '**', ext),
                                     recursive=True))
    cover_paths = sorted(list(dict.fromkeys(cover_paths)))

    for p in cover_paths:
        rel = os.path.relpath(p, cover_dir)
        img_id = rel.replace(os.sep, '_').rsplit('.', 1)[0]
        full_id = img_id
        cover_records.append({
            'cover_path': p,
            'image_id': full_id,
            'dataset': ds_name,
            'domain': domain
        })

    for alg in ALGOS:
        meta_path = os.path.join(BASE, ds_name, 'stego', alg, 'metadata.json')
        if not os.path.exists(meta_path):
            print(f"[{ds_name}] MISSING metadata: {meta_path}")
            continue

        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        for m in meta:
            if m.get('status') not in ('ok', 'warn_ber'):
                continue
            raw_id = m['image_id']
            full_id = raw_id
            spa = m.get('spatial_path', '')
            if spa.startswith('./'):
                spa = os.path.join(BASE, ds_name, spa[2:])
            spa = os.path.normpath(spa)
            stego_index[full_id][alg] = spa

    valid_cover_records = [
        r for r in cover_records
        if len(stego_index.get(r['image_id'], {})) > 0
    ]

    print(f"\n[{ds_name}] Index summary")
    print(f"  Cover records total        : {len(cover_records):,}")
    print(f"  Valid covers (with ≥1 stego): {len(valid_cover_records):,}")
    print(f"  Stego pairs total          : {sum(len(v) for v in stego_index.values()):,}")

    return valid_cover_records, stego_index


def build_index_task2(ds_name):
    """
    For Task 2: flatten all images (cover + each stego) with explicit labels.
    label = 0 for cover, 1..10 for algos.
    """
    cover_records, stego_index = build_index_for_dataset(ds_name)

    records = []
    # Cover images
    for rec in cover_records:
        records.append({
            'path': rec['cover_path'],
            'label': 0,
            'dataset': ds_name,
            'algo': 'cover'
        })

    # Stego images
    for rec in cover_records:
        img_id = rec['image_id']
        for alg, spa in stego_index[img_id].items():
            records.append({
                'path': spa,
                'label': ALGO_TO_ID[alg],
                'dataset': ds_name,
                'algo': alg
            })

    print(f"[{ds_name}] Task 2 dataset size: {len(records):,} "
          f"(covers={len(cover_records):,}, stegos≈{len(records)-len(cover_records):,})")
    return records

# %%
class BinaryStegoDataset(Dataset):
    """Task 1 per-dataset: random algo for stego pairs."""
    def __init__(self, records, stego_index, transform=None, random_algo=True):
        self.records = records
        self.stego_index = stego_index
        self.transform = transform
        self.random_algo = random_algo
        self.length = len(records) * 2

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rec = self.records[idx // 2]
        is_stego = (idx % 2 == 1)
        if is_stego:
            available = self.stego_index[rec['image_id']]
            algos = list(available.keys())
            alg = random.choice(algos) if self.random_algo else sorted(algos)[0]
            img_path = available[alg]
            label = 1.0
        else:
            img_path = rec['cover_path']
            label = 0.0

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new('RGB', IMAGE_SIZE, (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)


class AlgoMultiDataset(Dataset):
    """Task 2 per-dataset: full 11-class dataset."""
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img_path = r['path']
        label = r['label']
        algo = r['algo']

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new('RGB', IMAGE_SIZE, (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long), algo


# %%
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.05, contrast=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# %%
def make_splits_per_dataset(cover_records, save_dir):
    df = pd.DataFrame(cover_records)
    train_df, temp_df = train_test_split(
        df, test_size=0.20, random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED
    )

    train_df.to_csv(os.path.join(save_dir, 'task1_train_cover_split.csv'), index=False)
    val_df.to_csv(os.path.join(save_dir, 'task1_val_cover_split.csv'), index=False)
    test_df.to_csv(os.path.join(save_dir, 'task1_test_cover_split.csv'), index=False)

    print(f"  Train covers : {len(train_df):,} -> samples: {len(train_df)*2:,}")
    print(f"  Val covers   : {len(val_df):,} -> samples: {len(val_df)*2:,}")
    print(f"  Test covers  : {len(test_df):,} -> samples: {len(test_df)*2:,}")

    return (
        train_df.to_dict('records'),
        val_df.to_dict('records'),
        test_df.to_dict('records')
    )


def make_splits_task2(records, save_dir):
    df = pd.DataFrame(records)
    train_df, temp_df = train_test_split(
        df, test_size=0.20, random_state=SEED, stratify=df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED, stratify=temp_df['label']
    )

    train_df.to_csv(os.path.join(save_dir, 'task2_train_split.csv'), index=False)
    val_df.to_csv(os.path.join(save_dir, 'task2_val_split.csv'), index=False)
    test_df.to_csv(os.path.join(save_dir, 'task2_test_split.csv'), index=False)

    print(f"  Task2 train samples : {len(train_df):,}")
    print(f"  Task2 val samples   : {len(val_df):,}")
    print(f"  Task2 test samples  : {len(test_df):,}")

    return (
        train_df.to_dict('records'),
        val_df.to_dict('records'),
        test_df.to_dict('records')
    )

# %% [markdown]
# ## Helper Functions

# %%
def compute_binary_metrics(labels, preds, probs):
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    auc = roc_auc_score(labels, probs)
    cm = confusion_matrix(labels, preds)
    return acc, precision, recall, f1, auc, cm


def compute_multiclass_metrics(labels, preds):
    acc = accuracy_score(labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    macro_f1 = np.mean(f1)
    cm = confusion_matrix(labels, preds)
    return acc, macro_f1, precision, recall, f1, support, cm

# %% [markdown]
# ## Train--Eval. [Task-1]

# %%
def train_task1(model, train_loader, val_loader, device, save_dir, ds_name):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    best_auc = 0.0
    best_epoch = 0
    no_improve = 0
    patience = 7
    history = []

    start_time = time.time()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        pbar = tqdm(train_loader, desc=f"[{ds_name}][T1] Epoch {epoch}/{EPOCHS} (train)", ncols=100)
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == 'cuda')):
                logits = model(imgs).squeeze(1)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            running_loss += loss.item() * imgs.size(0)
            running_correct += (preds == labels.long()).sum().item()
            running_total += imgs.size(0)

            pbar.set_postfix(loss=loss.item())

        scheduler.step()

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        # Validation
        model.eval()
        val_loss_sum = 0.0
        all_probs = []
        all_labels = []

        pbar_val = tqdm(val_loader, desc=f"[{ds_name}][T1] Epoch {epoch}/{EPOCHS} (val)", ncols=100)
        with torch.no_grad():
            for imgs, labels in pbar_val:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(enabled=(device.type == 'cuda')):
                    logits = model(imgs).squeeze(1)
                    loss = criterion(logits, labels)

                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())
                val_loss_sum += loss.item() * imgs.size(0)
                pbar_val.set_postfix(loss=loss.item())

        val_loss = val_loss_sum / len(val_loader.dataset)
        val_probs = np.array(all_probs)
        val_labels = np.array(all_labels)
        val_preds = (val_probs >= 0.5).astype(int)

        val_acc, val_prec, val_rec, val_f1, val_auc, _ = compute_binary_metrics(
            val_labels, val_preds, val_probs
        )

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_prec': val_prec,
            'val_rec': val_rec,
            'val_f1': val_f1,
            'val_auc': val_auc
        })

        print(
            f"[{ds_name}][T1] Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}"
        )

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_auc': val_auc
        }, os.path.join(save_dir, 'task1_last_checkpoint.pth'))

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'task1_best_model.pth'))
            print(f"[{ds_name}][T1]  ✓ New best model saved (AUC={best_auc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{ds_name}][T1] Early stopping at epoch {epoch}")
                break

    elapsed_min = (time.time() - start_time) / 60.0
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == 'cuda' else 0.0

    pd.DataFrame(history).to_csv(os.path.join(save_dir, 'task1_training_history.csv'), index=False)

    print(f"[{ds_name}][T1] Best val AUC: {best_auc:.4f} at epoch {best_epoch}")
    print(f"[{ds_name}][T1] Training time: {elapsed_min:.2f} min, "
          f"Peak GPU memory: {peak_mem_gb:.2f} GB")

    return best_auc, elapsed_min, peak_mem_gb


def evaluate_task1(model, test_loader, device, save_dir, ds_name):
    model.eval()
    records = []

    pbar = tqdm(test_loader, desc=f"[{ds_name}][T1] Testing", ncols=100)
    with torch.no_grad():
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            with autocast(enabled=(device.type == 'cuda')):
                logits = model(imgs).squeeze(1)

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            for i in range(len(labels)):
                records.append({
                    'label': int(labels[i]),
                    'pred': int(preds[i]),
                    'prob': float(probs[i])
                })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(save_dir, 'task1_test_results.csv'), index=False)

    labels = df['label'].values
    preds = df['pred'].values
    probs = df['prob'].values

    acc, prec, rec, f1, auc, cm = compute_binary_metrics(labels, preds, probs)

    print(f"\n{'='*60}")
    print(f"{ds_name} — Task 1 Binary Steganalysis")
    print(f"{'='*60}")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Precision     : {prec:.4f}")
    print(f"Recall        : {rec:.4f}")
    print(f"F1-Score      : {f1:.4f}")
    print(f"AUC-ROC       : {auc:.4f}")
    print("Confusion Matrix (rows=gt, cols=pred) [Cover, Stego]:")
    print(cm)
    print()
    print(classification_report(labels, preds, target_names=['Cover', 'Stego']))
    print(f"{'='*60}\n")

    metrics = {
        'test_acc': acc,
        'test_prec': prec,
        'test_rec': rec,
        'test_f1': f1,
        'test_auc': auc,
        'confusion_matrix': cm.tolist()
    }
    with open(os.path.join(save_dir, 'task1_test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics

# %% [markdown]
# ## Train--Eval. [Task-2]

# %%
def train_task2(model, train_loader, val_loader, device, save_dir, ds_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    best_macro_f1 = 0.0
    best_epoch = 0
    no_improve = 0
    patience = 7
    history = []

    start_time = time.time()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        pbar = tqdm(train_loader, desc=f"[{ds_name}][T2] Epoch {epoch}/{EPOCHS} (train)", ncols=100)
        for imgs, labels, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == 'cuda')):
                logits = model(imgs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(logits, dim=1)
            running_loss += loss.item() * imgs.size(0)
            running_correct += (preds == labels).sum().item()
            running_total += imgs.size(0)

            pbar.set_postfix(loss=loss.item())

        scheduler.step()

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        # Validation
        model.eval()
        val_loss_sum = 0.0
        all_preds = []
        all_labels = []

        pbar_val = tqdm(val_loader, desc=f"[{ds_name}][T2] Epoch {epoch}/{EPOCHS} (val)", ncols=100)
        with torch.no_grad():
            for imgs, labels, _ in pbar_val:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(enabled=(device.type == 'cuda')):
                    logits = model(imgs)
                    loss = criterion(logits, labels)

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                val_loss_sum += loss.item() * imgs.size(0)
                pbar_val.set_postfix(loss=loss.item())

        val_loss = val_loss_sum / len(val_loader.dataset)
        val_labels = np.array(all_labels)
        val_preds = np.array(all_preds)
        val_acc, val_macro_f1, _, _, _, _, _ = compute_multiclass_metrics(
            val_labels, val_preds
        )

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_macro_f1': val_macro_f1
        })

        print(
            f"[{ds_name}][T2] Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f}"
        )

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_macro_f1': val_macro_f1
        }, os.path.join(save_dir, 'task2_last_checkpoint.pth'))

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'task2_best_model.pth'))
            print(f"[{ds_name}][T2]  ✓ New best model saved (Macro F1={best_macro_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{ds_name}][T2] Early stopping at epoch {epoch}")
                break

    elapsed_min = (time.time() - start_time) / 60.0
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == 'cuda' else 0.0

    pd.DataFrame(history).to_csv(os.path.join(save_dir, 'task2_training_history.csv'), index=False)

    print(f"[{ds_name}][T2] Best val Macro F1: {best_macro_f1:.4f} at epoch {best_epoch}")
    print(f"[{ds_name}][T2] Training time: {elapsed_min:.2f} min, "
          f"Peak GPU memory: {peak_mem_gb:.2f} GB")

    return best_macro_f1, elapsed_min, peak_mem_gb


def evaluate_task2(model, test_loader, device, save_dir, ds_name):
    model.eval()
    records = []

    pbar = tqdm(test_loader, desc=f"[{ds_name}][T2] Testing", ncols=100)
    with torch.no_grad():
        for imgs, labels, algos in pbar:
            imgs = imgs.to(device, non_blocking=True)
            with autocast(enabled=(device.type == 'cuda')):
                logits = model(imgs)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(len(labels_np)):
                records.append({
                    'label': int(labels_np[i]),
                    'pred': int(preds[i]),
                    'algo': algos[i]
                })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(save_dir, 'task2_test_results.csv'), index=False)

    labels = df['label'].values
    preds = df['pred'].values

    acc, macro_f1, prec, rec, f1, support, cm = compute_multiclass_metrics(labels, preds)

    print(f"\n{'='*60}")
    print(f"{ds_name} — Task 2 Algorithm Classification (11 classes)")
    print(f"{'='*60}")
    print(f"Accuracy           : {acc:.4f}")
    print(f"Macro F1-Score     : {macro_f1:.4f}")
    print("Confusion Matrix (rows=gt, cols=pred, classes 0..10):")
    print(cm)
    print()
    print("Per-class metrics:")
    for i in range(len(support)):
        name = 'cover' if i == 0 else ID_TO_ALGO.get(i, f'class_{i}')
        print(f"  {i:2d} ({name:10s}): "
              f"prec={prec[i]:.4f}, rec={rec[i]:.4f}, f1={f1[i]:.4f}, support={support[i]}")

    # Accuracy broken down by algorithm type (b)
    print("\nAccuracy per algorithm type:")
    algo_acc = {}
    for alg in ['cover'] + ALGOS:
        if alg == 'cover':
            mask = (df['label'] == 0)
        else:
            mask = (df['label'] == ALGO_TO_ID[alg])
        sub = df[mask]
        if len(sub) == 0:
            continue
        algo_acc[alg] = (sub['label'] == sub['pred']).mean()
        print(f"  {alg:10s}: acc={algo_acc[alg]:.4f} (n={len(sub)})")

    # Payload rate breakdown (c): placeholder until payload info is in metadata
    # Once you add a 'payload' column to df, you can group by it here.

    metrics = {
        'test_acc': acc,
        'test_macro_f1': macro_f1,
        'confusion_matrix': cm.tolist(),
        'per_class_precision': prec.tolist(),
        'per_class_recall': rec.tolist(),
        'per_class_f1': f1.tolist(),
        'per_class_support': support.tolist(),
        'per_algo_acc': algo_acc,
        'note_payload': 'Add payload metadata to compute per-payload accuracy.'
    }
    with open(os.path.join(save_dir, 'task2_test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"{'='*60}\n")
    return metrics

# %%
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_results_task1 = []
    all_results_task2 = []

    for ds_name in DATASETS.keys():
        print(f"\n\n{'#'*70}")
        print(f"Dataset: {ds_name}")
        print(f"{'#'*70}\n")

        save_dir = os.path.join(GLOBAL_SAVE_DIR, ds_name)
        os.makedirs(save_dir, exist_ok=True)

        # ===== Task 1 =====
        if RUN_TASK1:
            cover_records, stego_index = build_index_for_dataset(ds_name)
            if len(cover_records) < 10:
                print(f"[{ds_name}] Too few valid covers for Task 1, skipping.")
            else:
                train_recs, val_recs, test_recs = make_splits_per_dataset(cover_records, save_dir)

                train_ds = BinaryStegoDataset(train_recs, stego_index, transform=train_transform, random_algo=True)
                val_ds   = BinaryStegoDataset(val_recs,   stego_index, transform=eval_transform,  random_algo=False)
                test_ds  = BinaryStegoDataset(test_recs,  stego_index, transform=eval_transform,  random_algo=False)

                train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                          num_workers=NUM_WORKERS, pin_memory=True)
                val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE*2, shuffle=False,
                                          num_workers=NUM_WORKERS, pin_memory=True)
                test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE*2, shuffle=False,
                                          num_workers=NUM_WORKERS, pin_memory=True)

                print(f"[{ds_name}][T1] Train batches: {len(train_loader):,}")
                print(f"[{ds_name}][T1] Val batches  : {len(val_loader):,}")
                print(f"[{ds_name}][T1] Test batches : {len(test_loader):,}")

                model_t1 = ResNet18Binary().to(device)
                count_params(model_t1)
                best_auc, train_min, peak_gb = train_task1(
                    model_t1, train_loader, val_loader, device, save_dir, ds_name
                )

                model_t1.load_state_dict(torch.load(os.path.join(save_dir, 'task1_best_model.pth'),
                                                    map_location=device))
                test_metrics = evaluate_task1(model_t1, test_loader, device, save_dir, ds_name)

                all_results_task1.append({
                    'dataset': ds_name,
                    'best_val_auc': best_auc,
                    'test_auc': test_metrics['test_auc'],
                    'test_acc': test_metrics['test_acc'],
                    'test_f1': test_metrics['test_f1'],
                    'train_time_min': train_min,
                    'peak_gpu_gb': peak_gb
                })

        # ===== Task 2 =====
        if RUN_TASK2:
            records_t2 = build_index_task2(ds_name)
            if len(records_t2) < 100:  # sanity
                print(f"[{ds_name}] Too few samples for Task 2, skipping.")
            else:
                train_recs, val_recs, test_recs = make_splits_task2(records_t2, save_dir)

                train_ds2 = AlgoMultiDataset(train_recs, transform=train_transform)
                val_ds2   = AlgoMultiDataset(val_recs,   transform=eval_transform)
                test_ds2  = AlgoMultiDataset(test_recs,  transform=eval_transform)

                train_loader2 = DataLoader(train_ds2, batch_size=BATCH_SIZE, shuffle=True,
                                           num_workers=NUM_WORKERS, pin_memory=True)
                val_loader2   = DataLoader(val_ds2,   batch_size=BATCH_SIZE*2, shuffle=False,
                                           num_workers=NUM_WORKERS, pin_memory=True)
                test_loader2  = DataLoader(test_ds2,  batch_size=BATCH_SIZE*2, shuffle=False,
                                           num_workers=NUM_WORKERS, pin_memory=True)

                print(f"[{ds_name}][T2] Train batches: {len(train_loader2):,}")
                print(f"[{ds_name}][T2] Val batches  : {len(val_loader2):,}")
                print(f"[{ds_name}][T2] Test batches : {len(test_loader2):,}")

                model_t2 = ResNet18Multi(num_classes=11).to(device)
                count_params(model_t2)
                best_macro_f1, train_min2, peak_gb2 = train_task2(
                    model_t2, train_loader2, val_loader2, device, save_dir, ds_name
                )

                model_t2.load_state_dict(torch.load(os.path.join(save_dir, 'task2_best_model.pth'),
                                                    map_location=device))
                test_metrics2 = evaluate_task2(model_t2, test_loader2, device, save_dir, ds_name)

                all_results_task2.append({
                    'dataset': ds_name,
                    'best_val_macro_f1': best_macro_f1,
                    'test_macro_f1': test_metrics2['test_macro_f1'],
                    'test_acc': test_metrics2['test_acc'],
                    'train_time_min': train_min2,
                    'peak_gpu_gb': peak_gb2
                })

    if RUN_TASK1 and all_results_task1:
        df1 = pd.DataFrame(all_results_task1)
        df1.to_csv(os.path.join(GLOBAL_SAVE_DIR, 'summary_task1_per_dataset.csv'), index=False)
        print("Task 1 summary saved to summary_task1_per_dataset.csv")

    if RUN_TASK2 and all_results_task2:
        df2 = pd.DataFrame(all_results_task2)
        df2.to_csv(os.path.join(GLOBAL_SAVE_DIR, 'summary_task2_per_dataset.csv'), index=False)
        print("Task 2 summary saved to summary_task2_per_dataset.csv")


