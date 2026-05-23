install dependencies: sh dependencies
prep data: python add_masks.py
split data:
mac:
python build_datasets.py --CitrusUAT_dir data/CitrusUAT_dataset/preprocessed --OrangeLeaves_dir data/Orange_Leaves_Images_Dataset_for_the_Detection_of_Huanglongbing/Database_HLB_and_Healthy_-_Final --output_dir data/combined_split
windows:
python build_datasets.py --CitrusUAT_dir data\CitrusUAT_dataset\preprocessed --OrangeLeaves_dir data\Orange_Leaves_Images_Dataset_for_the_Detection_of_Huanglongbing\Database_HLB_and_Healthy_-_Final --output_dir data/combined_split