import argparse
import random
import os
from PIL import Image
from tqdm import tqdm   #progress bar for loops

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/images',
                    help="Directory with images dataset")
parser.add_argument('--output_dir', default='data/split_images',
                    help="Location of new data")

def get_class_name(filename):
    # print(filename)
    name = os.path.splitext(filename)[0]    #remove .jpg
    name = name.rsplit("_", 1)[0]   #remove number
    name = name.rsplit("\\", 1)[1]  #remove directory
    return name

def split_by_class(filename, output_dir):
    image = Image.open(filename)
    class_name = get_class_name(filename)
    output_path = os.path.join(output_dir, class_name)
    if not os.path.exists(output_path):
            os.mkdir(output_path)
    # print(os.path.join(output_path, os.path.basename(filename)))
    image.save(os.path.join(output_path, os.path.basename(filename)))

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find dataset at {}".format(args.data_dir)

    # Define the data directories
    train_data_dir = os.path.join(args.data_dir)
    test_data_dir = os.path.join(args.data_dir)

    # Get the filenames in each directory (train and test)
    filenames = os.listdir(train_data_dir)
    filenames = [os.path.join(train_data_dir, f) for f in filenames if f.endswith('.jpg')]

    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if f.endswith('.jpg')]

    # Split the images in 'train_signs' into 80% train and 20% dev
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(230)
    filenames.sort()
    random.shuffle(filenames)

    split = int(0.8 * len(filenames))
    train_filenames = filenames[:split]
    test_filenames = filenames[split:]

    filenames = {'train': train_filenames,
                 'test': test_filenames}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, dev and test
    for split in ['train', 'test']:
        output_dir_split = os.path.join(args.output_dir, split)
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            split_by_class(filename, output_dir_split)

    print("Done building dataset")