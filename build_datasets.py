import argparse
import random
import os
from PIL import Image
from tqdm import tqdm   #progress bar for loops

parser = argparse.ArgumentParser()
parser.add_argument('--CitrusUAT_dir', default = 'CitrusUAT/Images',
                    help="Directory to CitrusUAT dataset")
parser.add_argument('--OrangeLeaves_dir', default='Orange_leaves',
                    help="Directory to Orange Leaves Images")
parser.add_argument('--output_dir', default='data/split_images',
                    help="Location of new data")

def get_class_name(filename):
    name = os.path.basename(filename)
    return name.rsplit("_", 1)[0]

def split_by_class(filename, class_name, output_dir):
    image = Image.open(filename)
    output_path = os.path.join(output_dir, class_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    image.save(os.path.join(output_path, os.path.basename(filename)))

def gather_CitrusUAT_files(directory):
    if not os.path.isdir(directory):
        print(f"Warning: Directory not found {directory}")
        return []
    
    all_files = []
    for f in os.listdir(directory):
        if f.endswith('.jpg'):
            class_name = get_class_name(f)

            if class_name == "Healthy" or class_name == "HLB":
                full_path = os.path.join(directory, f)
                all_files.append((full_path, class_name))

    if len(all_files) == 0: 
        print(f"Warning: No images found in {directory}")
    
    return all_files

def gather_Orange_Leaves_files(directory):
    if not os.path.isdir(directory):
        print(f"Warning: Directory not found {directory}")
        return []
    
    all_files = []

    for class_folder in os.listdir(directory):
        class_path = os.path.join(directory, class_folder)
        if not os.path.isdir(class_path):
            continue

        images_folder = os.path.join(class_path, 'preprocessed_images')

        class_name = ""

        if "hlb" in class_folder.lower():
            class_name = "HLB"
        elif "healthy" in class_folder.lower():
            class_name = "Healthy"

        if os.path.isdir(images_folder):
            for f in os.listdir(images_folder):
                if f.endswith('.png'):
                    full_path = os.path.join(images_folder, f)
                    all_files.append((full_path, class_name))
    
    if len(all_files) == 0: 
        print(f"Warning: No images found in {directory}")
    
    return all_files

if __name__ == '__main__':
    args = parser.parse_args()

    all_data = []

    CitrusUAT_files = gather_CitrusUAT_files(args.CitrusUAT_dir)
    all_data.extend(CitrusUAT_files)

    Orange_leaves_files = gather_Orange_Leaves_files(args.OrangeLeaves_dir)
    all_data.extend(Orange_leaves_files)

    assert len(all_data) > 0, "No images found."

    random.seed(230)
    all_data.sort()
    random.shuffle(all_data)

    # 80/20 split for training set and test set
    split = int(0.8 * len(all_data))
    temp_train_filenames = all_data[:split]
    test_filenames = all_data[split:]

    # 80/20 split on training set for training and validation set
    val_split = int(0.8 * len(temp_train_filenames))
    train_filenames = temp_train_filenames[:val_split]
    val_filenames = temp_train_filenames[val_split:]


    filenames = {'train': train_filenames,
                 'val': val_filenames,
                 'test': test_filenames}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, dev and test
    for split in ['train', 'val', 'test']:
        output_dir_split = os.path.join(args.output_dir, split)
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        for filename, class_name in tqdm(filenames[split]):
            split_by_class(filename, class_name, output_dir_split)

    print("Done building dataset")