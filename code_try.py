# ===============================================================
# COMPLETE RT-DETR TRAINING + EVALUATION (YOLO-COMPARABLE)
# ===============================================================

import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm

from transformers import (
    RTDetrForObjectDetection,
    RTDetrImageProcessor,
    get_scheduler
)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ===============================================================
# CONFIGURATION
# ===============================================================
from roboflow import Roboflow
rf = Roboflow(api_key="gBDwOMgcuejZmxa38p9L")
project = rf.workspace("aktimuw").project("aktimuw")
version = project.version(4)
dataset = version.download("coco-mmdetection")

DATASET_ROOT = dataset.location         
print(f"Roboflow dataset extracted to: {DATASET_ROOT}")

IMG_SIZE = 640
BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-5
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================================================
# LOAD COCO DATASETS
# ===============================================================
def load_coco(split):
    ann_path = os.path.join(DATASET_ROOT, split, "_annotations.coco.json")
    coco = COCO(ann_path)

    images = []
    for img in coco.dataset["images"]:
        img_path = os.path.join(DATASET_ROOT, split, img["file_name"])
        images.append((img["id"], img_path))

    return coco, images


coco_train, train_imgs = load_coco("train")
coco_val, val_imgs = load_coco("valid")
coco_test, test_imgs = load_coco("test")

cat_ids = coco_train.getCatIds()
cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
idx_to_name = {i: coco_train.loadCats(cid)[0]["name"] for cid, i in cat_id_to_idx.items()}
NUM_CLASSES = len(cat_ids)

print(f"Number of classes: {NUM_CLASSES}")

# ===============================================================
# DATASET CLASS
# ===============================================================
class COCODataset(Dataset):
    def __init__(self, images, coco, processor):
        self.images = images
        self.coco = coco
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id, path = self.images[idx]
        image = Image.open(path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []

        for ann in anns:
            boxes.append(ann["bbox"])
            labels.append(cat_id_to_idx[ann["category_id"]])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        encoding = self.processor(images=image, return_tensors="pt")

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": {
                "boxes": boxes,
                "class_labels": labels
            },
            "image_id": img_id
        }

# ===============================================================
# COLLATE FUNCTION (CRITICAL)
# ===============================================================
def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch]).to(DEVICE)

    labels = []
    for b in batch:
        labels.append({
            "boxes": b["labels"]["boxes"].to(DEVICE),
            "class_labels": b["labels"]["class_labels"].to(DEVICE)
        })

    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "image_id": [b["image_id"] for b in batch]
    }

# ===============================================================
# MODEL & PROCESSOR
# ===============================================================
processor = RTDetrImageProcessor.from_pretrained(
    "PekingU/rtdetr_r50vd_coco_o365",
    do_resize=True,
    size={"height": IMG_SIZE, "width": IMG_SIZE},
    do_pad=True
)

model = RTDetrForObjectDetection.from_pretrained(
    "PekingU/rtdetr_r50vd_coco_o365",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
).to(DEVICE)

# ===============================================================
# DATALOADERS (WINDOWS SAFE)
# ===============================================================
train_ds = COCODataset(train_imgs, coco_train, processor)
val_ds = COCODataset(val_imgs, coco_val, processor)
test_ds = COCODataset(test_imgs, coco_test, processor)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=collate_fn, num_workers=0)

val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                    collate_fn=collate_fn, num_workers=0)

test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                     collate_fn=collate_fn, num_workers=0)

# ===============================================================
# COCO EVALUATION FUNCTION (YOLO-COMPARABLE)
# ===============================================================
def evaluate(model, dataloader, coco_gt):
    model.eval()
    coco_results = []

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(batch["pixel_values"])

            results = processor.post_process_object_detection(
                outputs,
                threshold=0.001,
                target_sizes=[(IMG_SIZE, IMG_SIZE)] * len(batch["image_id"])
            )

            for img_id, r in zip(batch["image_id"], results):
                for box, score, label in zip(r["boxes"], r["scores"], r["labels"]):
                    coco_results.append({
                        "image_id": img_id,
                        "category_id": int(label),
                        "bbox": [
                            float(box[0]),
                            float(box[1]),
                            float(box[2] - box[0]),
                            float(box[3] - box[1])
                        ],
                        "score": float(score)
                    })

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()

    return coco_eval.stats

# ===============================================================
# TRAINING + EPOCH-WISE VALIDATION
# ===============================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

scheduler = get_scheduler(
    "linear",
    optimizer,
    num_warmup_steps=100,
    num_training_steps=EPOCHS * len(train_dl)
)

results_log = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0

    for batch in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}"):
        optimizer.zero_grad()
        outputs = model(batch["pixel_values"], labels=batch["labels"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_dl)

    stats = evaluate(model, val_dl, coco_val)

    results_log.append({
        "epoch": epoch,
        "loss": avg_loss,
        "mAP@0.5": stats[1],
        "mAP@0.5:0.95": stats[0],
        "Recall": stats[8]
    })

    print(
        f"Epoch {epoch} | "
        f"Loss: {avg_loss:.4f} | "
        f"mAP@0.5: {stats[1]:.4f} | "
        f"mAP@0.5:0.95: {stats[0]:.4f}"
    )

# ===============================================================
# SAVE YOLO-STYLE RESULTS.CSV
# ===============================================================
df = pd.DataFrame(results_log)
df.to_csv(os.path.join(OUTPUT_DIR, "rtdetr_results.csv"), index=False)

print("\nSaved rtdetr_results.csv")

# ===============================================================
# FINAL TEST SET EVALUATION
# ===============================================================
final_stats = evaluate(model, test_dl, coco_test)

print("\nFINAL TEST METRICS")
print(f"mAP@0.5:0.95 = {final_stats[0]:.4f}")
print(f"mAP@0.5      = {final_stats[1]:.4f}")
print(f"Recall       = {final_stats[8]:.4f}")

print("\nRT-DETR TRAINING + EVALUATION COMPLETED SUCCESSFULLY 🎉")
