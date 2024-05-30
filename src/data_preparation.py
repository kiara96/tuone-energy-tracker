import pandas as pd
import json
from sklearn.model_selection import train_test_split
from datasets import Dataset
import config
from config import DATA_FILE_PATH

def load_data(file_path):
    dataframes = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            df = pd.json_normalize(json_obj)
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

def transform_to_dict(row):
    tags = [{'start': span['start'], 'end': span['end'], 'tag': span['label']} for span in row['spans']] if isinstance(row['spans'], list) else []
    return {'tags': tags, 'id': row['meta.ID'], 'text': row['text']}


def prepare_datasets():
    df = load_data(config.DATA_FILE_PATH)
    print(f"Loaded dataset size: {len(df)}")  # Check size
    new_data = df.apply(transform_to_dict, axis=1)
    train_data, val_data = train_test_split(new_data, test_size=0.2, random_state=42)
    train_data.to_json('data/annotations.test.train.jsonlines', lines=True, orient='records')
    val_data.to_json('data/annotations.test.validation.jsonlines', lines=True, orient='records')

    def load_jsonl_to_df(path):
        return pd.read_json(path, lines=True)

    train_data = load_jsonl_to_df('data/annotations.test.train.jsonlines')
    val_data = load_jsonl_to_df('data/annotations.test.validation.jsonlines')

    # Convert DataFrame to Dataset
    train_ds = Dataset.from_pandas(train_data)
    val_ds = Dataset.from_pandas(val_data)
    print(f"Training Dataset Size: {len(train_ds)}")  # Output the size of the training dataset
    print(f"Validation Dataset Size: {len(val_ds)}")  # Output the size of the validation dataset
    # for i in range(3):
    #     example = train_ds[i]
    #     print(f"\n{example['text']}")
    #     for tag_item in example["tags"]:
    #         print(tag_item["tag"].ljust(10), "-", example["text"][tag_item["start"]: tag_item["end"]])
    return train_ds, val_ds
