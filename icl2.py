# ICL - BEST/WORST

import random as rand
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import re
import gc
import os
from PIL import Image
from collections import defaultdict
from torchvision import datasets
from torch.utils.data import Subset, random_split
import transformers
from transformers import AutoModel, AutoImageProcessor, Trainer, TrainingArguments, TrainerCallback, AutoProcessor, AutoModelForMultimodalLM
from sklearn.metrics import f1_score

# split 
train_path = os.path.join("data", "combined_split", "train")
val_path = os.path.join("data", "combined_split", "val")
test_path = os.path.join("data", "combined_split", "test")

# ================================
# DINOv3 FINE-TUNING
# ================================

# dataset wrapper
class dinodataset():
    def __init__(self, root, processor):
        self.ds = datasets.ImageFolder(root)
        self.processor = processor
        self.targets = self.ds.targets
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        img, label = self.ds[idx]
        inputs = self.processor(images=img, return_tensors="pt")
        return {"pixel_values": inputs["pixel_values"].squeeze(0), "labels": torch.tensor(label)}

num_classes = len(os.listdir(train_path))

# custom -> wrap backbone and head
class DINOClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.config.hidden_size, num_classes)

    def forward(self, pixel_values, labels=None):
        out = self.backbone(pixel_values=pixel_values)
        logits = self.classifier(out.last_hidden_state[:, 0])
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    f1 = f1_score(labels, preds, average="macro")
    acc = (preds == labels).mean()
    return {"f1": f1, "acc": acc}

# print per epoch while training
class EpochMetricsCallback(TrainerCallback):
    def __init__(self):
        self.metrics = {}

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self.metrics.update(metrics)

        if "eval_train_f1" in self.metrics and "eval_val_f1" in self.metrics:
            self.metrics = {}

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dinov3_model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"

dinov3_processor = AutoImageProcessor.from_pretrained(dinov3_model_name)
dinov3_backbone = AutoModel.from_pretrained(dinov3_model_name).to(device)
dinov3 = DINOClassifier(dinov3_backbone, num_classes)

train_ds = dinodataset(train_path, dinov3_processor)
val_ds = dinodataset(val_path, dinov3_processor)

lr = 1e-5
num_epochs = 9
training_args = TrainingArguments(
    output_dir="./dinooutput",
    eval_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=num_epochs,
    learning_rate=lr,
    save_steps=500,
    save_total_limit=2,
)

dinov3_trainer = Trainer(
    model=dinov3,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset={"train": train_ds, "val": val_ds},
    compute_metrics=compute_metrics,
    callbacks=[EpochMetricsCallback()],
)

dinov3_trainer.train()

# rank validation set
classes = val_ds.ds.classes

dinov3 = dinov3.eval()

def infer(image, device="cpu"):
    with torch.no_grad():
        inputs = dinov3_processor(images=image, return_tensors="pt").to(device)
        logits = dinov3(inputs["pixel_values"])["logits"]
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1).item()
        conf = probs[0, pred].item()
        pred_class = classes[pred]
        return pred_class, conf

val_data = []

labels = ['Healthy', 'HLB']
for label in labels:
    path = os.path.join(val_path, label)
    for f in os.listdir(path):
        img_path = os.path.join(path, f)
        val_data.append((img_path, label))

scored_data = []
for img_path, label in val_data:
    img = Image.open(img_path)
    pred, conf = infer(img, device)
    if pred != label:
        conf = 1 - conf
    scored_data.append((conf, img_path, label))

# clean up
del dinov3_processor, dinov3_backbone, dinov3, dinov3_trainer
del train_ds, val_ds
gc.collect()

# ================================
# ICL
# ================================

def load_image(path):
    return Image.open(path).convert("RGB")

def generate_message(query_img, prompt, icl_pairs=[]):
    content = [{"type": "text", "text": prompt}]

    for _, img_path, label in icl_pairs:
        content.append({"type": "image", "image": load_image(img_path)})
        content.append({"type": "text",  "text": label})

    content.append({"type": "image", "image": load_image(query_img)})

    return [{"role": "user", "content": content}]

# sampling
def subset(dataset, frac, seed=None):
    strata = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        strata[label].append(idx)

    subset_i = []
    for label, idxs in sorted(strata.items()):
        n = max(1, int(len(idxs) * frac))
        if seed:
            perm = torch.randperm(len(idxs), generator=torch.Generator().manual_seed(int(seed)))
        else:
            perm = torch.randperm(len(idxs))
        sample = [idxs[i] for i in perm[:n]]
        subset_i.extend(sample)

    rand.shuffle(subset_i)
    
    return Subset(dataset, subset_i)

MODEL_ID = "google/gemma-4-E2B-it"

# load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID, 
    dtype="auto",
).to("cpu")

test_data = []

labels = ['Healthy', 'HLB']
for label in labels:
    path = os.path.join(test_path, label)
    for f in os.listdir(path):
        img_path = os.path.join(path, f)
        test_data.append((img_path, label))

# shuffle datasets
rand.seed(8)
test_data = subset(test_data, 0.25)

prompt_0 = ("Label this leaf image as being infected with Huanglongbing disease or healthy. " 
                "Strictly use the labels 'HLB' and 'Healthy'. Provide no explanation and do not address any nuance.")

prompt_no_context = ("Label this leaf image as being infected with Huanglongbing disease or healthy. " 
                    "Strictly use the labels 'HLB' and 'Healthy'. "
                    "Given the following example(s), label only the last one image to the best of your ability.")

prompt_context = ("Huanglongbing disease is a disease of citrus trees distinguished by asymmetrical yellowing "
                "of the veins and adjacent tissues. Leaves contain splotchy mottling. "
                "Label this leaf image as being infected with Huanglongbing disease or healthy. " 
                "Strictly use the labels “HLB” and “Healthy”. Given the following example(s), label only the last one image to the best of your ability.")

# parse from csv
def parse_nums(value):
    nums = re.findall(r"[\d.]+(?:e[+-]?\d+)?", value)
    return np.array([int(n) for n in nums])

seeds_path = os.path.join('notebooks', 'runseeds.csv')
seeds = pd.read_csv(seeds_path)
seeds['Seed'] = seeds['Seed'].apply(
    lambda x: parse_nums(x) if isinstance(x, str) else x
)

# sampling, no sort
def subset(dataset, frac, seed=None):
    strata = defaultdict(list)
    for idx, (_, _, label) in enumerate(dataset):
        strata[label].append(idx)

    subset_i = []
    for label, idxs in sorted(strata.items()):
        n = max(1, int(len(idxs) * frac))
        if seed:
            perm = torch.randperm(len(idxs), generator=torch.Generator().manual_seed(int(seed)))
        else:
            perm = torch.randperm(len(idxs))
        sample = [idxs[i] for i in perm[:n]]
        subset_i.extend(sample)
    
    return Subset(dataset, subset_i)

fracs = [0.02, 0.07, 0.17, 0.35, 0.7, 1] # corresponds to train 1%, 2%, 5%, 10%, 20%, 30% (ish)

# ================================
# SAMPLE BEST EXAMPLES
# ================================

os.makedirs('resultsbest', exist_ok=True)

rng = rand.Random()
new_seeds = []
results = []

scored_data.sort(reverse=True)

for p_type, prompt in [("no-context", prompt_no_context), ("context", prompt_context)]: 
    print(f"{p_type}...")

    for i in range(len(fracs)):
        frac = fracs[i]
        print(f"    {frac * 100}%...")

        curr_seeds = []
        curr_res = []

        for run in range(5):
            y_pred = []
            y_actual = []
            print(f"        run {run + 1}")

            if p_type == "no-context":
                seed = rng.randint(0, 2**32 - 1)
                curr_seeds.append(seed)
            else:
                seed = new_seeds[i]['Seed'][run]
            icl_pairs = subset(scored_data, frac, seed)
            print(f"    examples: {len(icl_pairs)}")
            fname = f'{p_type}_{frac}_{run}.csv'
                        
            with open(os.path.join('resultsbest', fname), 'w') as f:
                for img_path, label in test_data:
                    messages = generate_message(img_path, prompt, icl_pairs)

                    inputs = processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        add_generation_prompt=True,
                    ).to(model.device)
                    input_len = inputs["input_ids"].shape[-1]

                    while(True):
                        outputs = model.generate(**inputs, max_new_tokens=16)
                        response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

                        parsed = processor.parse_response(response)
                        
                        pred = parsed['content']
                        if pred == "HLB" or pred == "Healthy": break

                    print(f"Image: {img_path}, Prediction: {pred}, Result: {label}")
                    f.write(f"Image: {img_path}, Prediction: {pred}, Result: {label}\n")

                    y_pred.append(pred)
                    y_actual.append(label)

                    del messages, inputs, outputs, response
                    gc.collect()
                    
            curr_res.append(f1_score(y_actual, y_pred, pos_label='HLB'))

        results.append({"Run": f"{p_type} {frac*100}%", "F1 Score": curr_res})

        if p_type == "no-context":
            new_seeds.append({"Percentage": f"{frac * 100}%", "Seed": curr_seeds})
        
df_seeds = pd.DataFrame(new_seeds)
df_seeds['Seed'] = df_seeds['Seed'].apply(lambda x: f"[{' '.join(map(str, x))}]")
df_seeds.to_csv('runseedsbw.csv', index=False)

df = pd.DataFrame(results)
df['F1 Score'] = df['F1 Score'].apply(lambda x: f"[{' '.join(map(str, x))}]")  # convert list to space separated instead of comma separated
df.to_csv('iclbestf1.csv', index=False)

# ================================
# SAMPLE WORST EXAMPLES
# ================================

os.makedirs('resultsworst', exist_ok=True)

results = []

scored_data.sort()

for p_type, prompt in [("no-context", prompt_no_context), ("context", prompt_context)]: 
    print(f"{p_type}...")

    for i in range(len(fracs)):
        frac = fracs[i]
        print(f"    {frac * 100}%...")

        curr_res = []

        for run in range(5):
            y_pred = []
            y_actual = []
            print(f"        run {run + 1}")

            seed = new_seeds[i]['Seed'][run]
            icl_pairs = subset(scored_data, frac, seed)
            print(f"    examples: {len(icl_pairs)}")
            fname = f'{p_type}_{frac}_{run}.csv'
                        
            with open(os.path.join('resultsworst', fname), 'w') as f:
                for img_path, label in test_data:
                    messages = generate_message(img_path, prompt, icl_pairs)

                    inputs = processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        add_generation_prompt=True,
                    ).to(model.device)
                    input_len = inputs["input_ids"].shape[-1]

                    outputs = model.generate(**inputs, max_new_tokens=16)
                    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

                    parsed = processor.parse_response(response)
                    
                    pred = parsed['content']

                    print(f"Image: {img_path}, Prediction: {pred}, Result: {label}")
                    f.write(f"Image: {img_path}, Prediction: {pred}, Result: {label}\n")

                    y_pred.append(pred)
                    y_actual.append(label)

                    del messages, inputs, outputs, response
                    gc.collect()

            curr_res.append(f1_score(y_actual, y_pred, pos_label='HLB'))

        results.append({"Run": f"{p_type} {frac*100}%", "F1 Score": curr_res})

df = pd.DataFrame(results)
df['F1 Score'] = df['F1 Score'].apply(lambda x: f"[{' '.join(map(str, x))}]")  # convert list to space separated instead of comma separated
df.to_csv('iclworstf1.csv', index=False)