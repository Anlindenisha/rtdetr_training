# --------------------------------------------------------------
#  RT-DETR TRAINING SCRIPT – GITHUB + ROBOFLOW READY
# --------------------------------------------------------------

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, get_scheduler
from tqdm.auto import tqdm
import roboflow
import json

# --------------------- CONFIG ---------------------
BATCH_SIZE   = 4
EPOCHS       = 10
LR           = 1e-5
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 100
GRAD_CLIP    = 1.0
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR   = "./rtdetr_finetuned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------- DOWNLOAD DATASET --------------------- 

from roboflow import Roboflow
rf = Roboflow(api_key="gBDwOMgcuejZmxa38p9L")
project = rf.workspace("aktimuw").project("aktimuw")
version = project.version(4)
dataset = version.download("coco-mmdetection")

DATASET_ROOT = dataset.location         
print(f"Roboflow dataset extracted to: {DATASET_ROOT}")
                

# --------------------- LOAD COCO SPLITS ---------------------
def load_coco_split(split: str):
    ann_path = os.path.join(DATASET_ROOT, split, "_annotations.coco.json")
    data = json.load(open(ann_path))
    img_map = {img["id"]: img for img in data["images"]}
    ann_map = {}
    for ann in data["annotations"]:
        ann_map.setdefault(ann["image_id"], []).append(ann)

    img_paths, img_ids, orig_ids = [], [], []
    for iid, info in img_map.items():
        path = os.path.join(DATASET_ROOT, split, info["file_name"])
        img_paths.append(path)
        img_ids.append(iid)
        orig_ids.append(info["id"])

    return img_paths, img_ids, ann_map, data["categories"], orig_ids

train_paths, train_ids, train_ann, cats, train_orig = load_coco_split("train")
val_paths, val_ids, val_ann, _, val_orig = load_coco_split("valid")
test_paths, test_ids, test_ann, _, test_orig = load_coco_split("test")

cat_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(cats)}
idx_to_name = {idx: cat["name"] for idx, cat in enumerate(cats)}
NUM_LABELS = len(cats)

print(f"Classes: {NUM_LABELS} -> {[c['name'] for c in cats]}")

# --------------------- MODEL & PROCESSOR ---------------------
processor = RTDetrImageProcessor.from_pretrained(
    "PekingU/rtdetr_r50vd_coco_o365",
    do_resize=True,
    size={"height": 640, "width": 640},
    do_pad=True,
    size_divisor=32,
)

model = RTDetrForObjectDetection.from_pretrained(
    "PekingU/rtdetr_r50vd_coco_o365",
    num_labels=NUM_LABELS,
    ignore_mismatched_sizes=True,
).to(DEVICE)

# --------------------- DATASET CLASS ---------------------
class COCODataset(Dataset):
    def __init__(self, img_paths, img_ids, ann_map, processor, cat_map, orig_ids):
        self.img_paths = img_paths
        self.img_ids = img_ids
        self.ann_map = ann_map
        self.processor = processor
        self.cat_map = cat_map
        self.orig_ids = orig_ids

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        iid = self.img_ids[idx]
        img = Image.open(path).convert("RGB")
        raw_anns = self.ann_map.get(iid, [])

        if raw_anns:
            class_labels = torch.tensor([self.cat_map[a["category_id"]] for a in raw_anns], dtype=torch.long)
            boxes = torch.tensor([a["bbox"] for a in raw_anns], dtype=torch.float)
        else:
            class_labels = torch.empty((0,), dtype=torch.long)
            boxes = torch.empty((0, 4), dtype=torch.float)

        labels = {"class_labels": class_labels, "boxes": boxes}
        enc = self.processor(images=img, return_tensors="pt")
        pixel_values = enc["pixel_values"].squeeze(0)

        return {"pixel_values": pixel_values, "labels": labels, "image_id": self.orig_ids[idx]}

def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch]).to(DEVICE)
    labels = [{"class_labels": b["labels"]["class_labels"].to(DEVICE),
               "boxes": b["labels"]["boxes"].to(DEVICE)} for b in batch]
    return {"pixel_values": pixel_values, "labels": labels, "image_id": [b["image_id"] for b in batch]}

train_ds = COCODataset(train_paths, train_ids, train_ann, processor, cat_id_to_idx, train_orig)
val_ds = COCODataset(val_paths, val_ids, val_ann, processor, cat_id_to_idx, val_orig)
test_ds = COCODataset(test_paths, test_ids, test_ann, processor, cat_id_to_idx, test_orig)

# --------------------- MAIN TRAINING ---------------------
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    num_steps = len(train_dl) * EPOCHS
    lr_scheduler = get_scheduler("linear", optimizer, WARMUP_STEPS, num_steps)
    scaler = torch.amp.GradScaler()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in tqdm(train_dl, desc=f"Epoch {epoch}"):
            px = batch["pixel_values"]
            labels = batch["labels"]

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                out = model(px, labels=labels)
                loss = out.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

        print(f"\nEpoch {epoch} complete (loss={loss.item():.4f})")

    print("\nTraining complete!")

 # --------------------- SAFE SAVE (Standard PyTorch & HF) ---------------------
    print("Training complete! Saving model safely...")
    
    # 1. Standard PyTorch State Dict Save (.pth)
    # This is the single file you will download for local JIT conversion.
    save_path = os.path.join(OUTPUT_DIR, "rtdetr_final.pth")
    torch.save(model.state_dict(), save_path)
    print(f" -> Weights saved to {save_path}")

    # 2. Hugging Face Pretrained Save (Folder)
    # This saves the configuration, making it easy to reload later.
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f" -> Hugging Face model and processor saved to {OUTPUT_DIR}")

    print("All saving tasks completed successfully.")
