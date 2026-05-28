import cv2
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import gc
import os
# import math
import shutil
from PIL import Image 

def load_image(path):
    return Image.open(path).convert("RGB")

# load model
model_name = "Qwen/Qwen2.5-VL-32B-Instruct"

min_pixels = 384 * 384
max_pixels = 768 * 768


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(
    model_name,
    min_pixels=min_pixels,
    max_pixels=max_pixels
)

train_path = "data/combined_split/train"
train_healthy = "data/combined_split/test/Healthy"
train_hlb = "data/combined_split/test/HLB"

train_data = []

labels = ['Healthy', 'HLB']
for label in labels:
    path = os.path.join(train_path, label)
    for f in os.listdir(path):
        img_path = os.path.join(path, f)
        # img = cv2.imread(img_path)
        train_data.append((img_path, label))

# print(train_data)

test_data = []

prompt_0 = ("Label this leaf image as being infected with Huanglongbing disease or healthy. " 
                "Use the labels 'HLB' and 'Healthy'.")

prompt_no_context = ("Label this leaf image as being infected with Huanglongbing disease or healthy. " 
                    "Use the labels 'HLB' and 'Healthy'. "
                    "Given the following example(s), label the last image to the best of your ability.")

prompt_context = ("Huanglongbing disease is a disease of citrus trees distinguished by asymmetrical yellowing "
                "of the veins and adjacent tissues. Leaves contain splotchy mottling. "
                "Label this leaf image as being infected with Huanglongbing disease or healthy. " 
                "Use the labels “HLB” and “Healthy”. Given the following example(s), label the last image to the best of your ability.")

# message format: prompt + input/output pairs + query image
def generate_message(query_img, prompt, icl_pairs=[]):
    content = [{"type": "text", "text": prompt}]

    for img_path, label in icl_pairs:
        content.append({"type": "image", "image": load_image(img_path)})
        content.append({"type": "text",  "text": label})

    content.append({"type": "image", "image": query_img})

    return [{"role": "user", "content": content}]

train_images = []
train_labels = []
subset = []

for p_type, prompt in [("zero-shot", prompt_0), ("no_context", prompt_no_context), ("context", prompt_context)]: 

    for percent in subset:
        icl_pairs = []  # USE SEED TO COPY ICL PAIRS HERE

    with open(f'results_{p_type}_{percent}shot.txt', 'w') as f:
        for img_path, label in test_data:
            query_img = load_image(img_path)

            # Generate message
            messages = generate_message(query_img, prompt, icl_pairs)

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            #Process video and image input
            image_inputs, video_inputs = process_vision_info(messages)

            #Preprocess inputs
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )


            max_input_length = 32768
            if inputs["input_ids"].shape[1] > max_input_length:
                inputs["input_ids"] = inputs["input_ids"][:, :max_input_length]
                inputs["attention_mask"] = inputs["attention_mask"][:, :max_input_length]

            inputs = {
                k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=64, 
                do_sample=False,
                use_cache=False
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]

            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            print(f"Image: {img_path}, Result: {str(output_text[0])}")
            f.write(f"{img_path}: {str(output_text[0])}\n")

            del inputs, image_inputs, video_inputs, generated_ids, generated_ids_trimmed, text, output_text, messages_no_ICL #change ICL here
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()