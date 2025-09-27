import os
import re
import pandas as pd
from tqdm import tqdm

class DataLoader:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        print(f"DataLoader initialized for path: {self.dataset_path}")

    def load_and_structure(self) -> pd.DataFrame:
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"The specified dataset path does not exist: {self.dataset_path}"
            )

        data = []
        
        # Define the directories to iterate over
        dir_map = {
            'train/pos': 1, 'train/neg': 0,
            'test/pos': 1, 'test/neg': 0
        }

        for sub_dir, label in dir_map.items():
            path = os.path.join(self.dataset_path, sub_dir)
            
            print(f"Loading reviews from: {path}")
            
            # Use tqdm for a progress bar over the files in each directory
            for filename in tqdm(os.listdir(path), desc=f"Processing {sub_dir}"):
                if filename.endswith(".txt"):
                    file_path = os.path.join(path, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        review_text = f.read()

                    # Use regex to parse star rating from filename (e.g., "123_9.txt")
                    match = re.search(r'_(\d+)\.txt$', filename)
                    star_rating = int(match.group(1)) if match else -1
                    
                    # Determine if the file is from the train or test set
                    split = 'train' if 'train' in sub_dir else 'test'

                    data.append({
                        "review_text": review_text,
                        "sentiment_label": label,
                        "star_rating": star_rating,
                        "split": split
                    })

        print("Data loading complete. Creating DataFrame.")
        return pd.DataFrame(data)

