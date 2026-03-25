install dependencies: sh dependencies
prep data: python add_masks.py
split data:
python build_datasets.py --data_dir data/CitrusUAT_dataset/preprocessed --output_dir data/CitrusUAT_split_images