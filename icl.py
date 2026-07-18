# ICL - RANDOM

import random as rand
import torch
import gc
import os
from PIL import Image
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import Subset
from transformers import AutoProcessor, AutoModelForMultimodalLM
from sklearn.metrics import f1_score

def load_image(path):
    return Image.open(path).convert("RGB")

def generate_message(query_img, prompt, icl_pairs=[]):
    content = [{"type": "text", "text": prompt}]

    for img_path, label in icl_pairs:
        content.append({"type": "image", "image": load_image(img_path)})
        content.append({"type": "text",  "text": label})

    content.append({"type": "image", "image": load_image(query_img)})

    return [{"role": "user", "content": content}]

# sampling
def subset(dataset, frac, seed=None):
    strata = defaultdict(list)
    for idx, (img_path, label) in enumerate(dataset):
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

train_path = os.path.join("data", "combined_split", "train")
train_data = []

labels = ['Healthy', 'HLB']
for label in labels:
    path = os.path.join(train_path, label)
    for f in os.listdir(path):
        img_path = os.path.join(path, f)
        train_data.append((img_path, label))

test_path = os.path.join("data", "combined_split", "test")
test_data = []

for label in labels:
    path = os.path.join(test_path, label)
    for f in os.listdir(path):
        img_path = os.path.join(path, f)
        test_data.append((img_path, label))

rand.seed(8)

# subset of test set
test_data = subset(test_data, 0.25) # 29

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

fracs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5] # RESOURCE LIMIT: 70%

offset = 2 # new percentages (not in fine-tuning)

os.makedirs('results', exist_ok=True)

rng = rand.Random()
new_seeds = []

results = []

for p_type, prompt in [("zero-shot", prompt_0), ("no-context", prompt_no_context), ("context", prompt_context)]: 
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
            if p_type == "zero-shot":
                icl_pairs = []
                fname = f'{p_type}.csv'
            else:
                if frac >= 0.05:
                    seed = seeds['Seed'].iloc[i - offset][run]
                elif p_type == "no-context":
                    seed = rng.randint(0, 2**32 - 1)
                    curr_seeds.append(seed)
                else:
                    seed = new_seeds[i]['Seed'][run]
                icl_pairs = subset(train_data, frac, seed)
                print(f"    examples: {len(icl_pairs)}")
                fname = f'{p_type}_{frac}.csv'
                        
            with open(os.path.join('results', fname), 'w') as f:
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

        if p_type == "zero-shot": 
            results.append({"Run": p_type, "F1 Score": curr_res})
            break
        else:
            results.append({"Run": f"{p_type} {frac*100}%", "F1 Score": curr_res})

        if frac < 0.05 and p_type == "no-context":
            new_seeds.append({"Percentage": f"{frac * 100}%", "Seed": curr_seeds})

df_seeds = pd.DataFrame(new_seeds)
df_seeds['Seed'] = df_seeds['Seed'].apply(lambda x: f"[{' '.join(map(str, x))}]")
df_seeds.to_csv('addlrunseeds.csv', index=False)

df = pd.DataFrame(results)
df['F1 Score'] = df['F1 Score'].apply(lambda x: f"[{' '.join(map(str, x))}]")  # convert list to space separated instead of comma separated
df.to_csv('iclf1.csv', index=False)