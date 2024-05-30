# config.py

# Model and data configuration parameters
MAX_LENGTH = 256
MODEL_PATH = "bert-base-uncased"
DATA_FILE_PATH = "/data/tuone_labelling.jsonl" #"./data/tuone_labelling.jsonl"
TRAIN_FILE = "data/annotations.train.jsonlines"
VALIDATION_FILE = "data/annotations.validation.jsonlines"
OUTPUT_DIR = "./models/fine_tune_bert_output"

# Tag to ID mapping
tag2id = {'ORG': 1, 'TECH': 2, 'LOC': 3, 'STATUS': 4, 'CAPACITY': 5, 'VALUE': 6, 'SUBSIDY': 7, 'JOBS': 8}
id2tag = {v: k for k, v in tag2id.items()}

# Label to ID mapping for "IOB" tagging scheme
label2id = {
    'O': 0,
    **{f'B-{k}': 2*v - 1 for k, v in tag2id.items()},
    **{f'I-{k}': 2*v for k, v in tag2id.items()}
}
id2label = {v: k for k, v in label2id.items()}
