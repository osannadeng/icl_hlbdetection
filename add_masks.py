import numpy as np
import cv2
from PIL import Image
import os


def add_mask(img_name, mask_name):
    img = cv2.imread(os.path.join(imgs_path, img_name))
    mask = cv2.imread(os.path.join(masks_path, mask_name), cv2.IMREAD_GRAYSCALE)
    return cv2.bitwise_and(img, img, mask=mask)

def get_mask_name(img):
    name_seg = img.replace(".jpg", "")
    return name_seg + "_mask.png"

# apply to CitrusUAT dataset
ds_path = os.path.join("data", "CitrusUAT_dataset")
imgs_path = os.path.join(ds_path, "Images")
masks_path = os.path.join(ds_path, "Masks")

for root, dirs, imgs in os.walk(imgs_path):
    for img in imgs:
        mask = get_mask_name(img)
        res = add_mask(img, mask)

        # add to folder
        res_path = os.path.join(ds_path, "preprocessed", img)
        cv2.imwrite(res_path, res)