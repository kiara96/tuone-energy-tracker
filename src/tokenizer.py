### MULTILABELLING
# from transformers import RobertaTokenizerFast
# from config import label2id

# # Initialize the tokenizer
# tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

# def get_token_role_in_span(token_start: int, token_end: int, span_start: int, span_end: int):
#     if token_end <= token_start:
#         return "N"
#     if token_start < span_start or token_end > span_end:
#         return "O"
#     if token_start >= span_start and token_end <= span_end:
#         return "I"
#     if token_start == span_start:
#         return "B"
#     return "O"

# MAX_LENGTH = 256

# def tokenize_and_adjust_labels(sample):
#     """
#     Args:
#         - sample (dict): {"id": "...", "text": "...", "tags": [{"start": ..., "end": ..., "tag": ...}, ...]
#     Returns:
#         - The tokenized version of `sample` and the labels of each token.
#     """
#     # Tokenize the text, keep the start and end positions of tokens with `return_offsets_mapping` option
#     # Use max_length and truncation to ajust the text length
#     tokenized = tokenizer(sample["text"], 
#                           return_offsets_mapping=True, 
#                           padding="max_length", 
#                           max_length=MAX_LENGTH,
#                           truncation=True)
    
#     if 'offset_mapping' not in tokenized:
#         raise KeyError("offset_mapping not found in tokenized output")
    
#     labels = [[0 for _ in range(len(label2id) * 2)] for _ in range(len(tokenized["input_ids"]))]
    
#     for i, (token_start, token_end) in enumerate(tokenized["offset_mapping"]):
#         for span in sample["tags"]:
#             role = get_token_role_in_span(token_start, token_end, span["start"], span["end"])
#             if role == "B":
#                 labels[i][label2id[f"B-{span['tag']}"]] = 1
#             elif role == "I":
#                 labels[i][label2id[f"I-{span['tag']}"]] = 1

#     tokenized.pop("offset_mapping")

#     # Debugging statements
#     print(f"Sample text: {sample['text']}")
#     print(f"Tokenized text: {tokenizer.convert_ids_to_tokens(tokenized['input_ids'])}")
#     print(f"Tags: {sample['tags']}")
#     print(f"Labels: {labels}")
    
#     return {**tokenized, "labels": labels}

from transformers import RobertaTokenizerFast
from config import label2id

# Initialize the tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

def get_token_role_in_span(token_start: int, token_end: int, span_start: int, span_end: int):
    if token_end <= token_start:
        return "N"
    if token_start < span_start or token_end > span_end:
        return "O"
    if token_start >= span_start and token_end <= span_end:
        return "I"
    if token_start == span_start:
        return "B"
    return "O"

MAX_LENGTH = 256

def tokenize_and_adjust_labels(sample):
    """
    Args:
        - sample (dict): {"id": "...", "text": "...", "tags": [{"start": ..., "end": ..., "tag": ...}, ...]
    Returns:
        - The tokenized version of `sample` and the labels of each token.
    """
    # Tokenize the text, keep the start and end positions of tokens with `return_offsets_mapping` option
    # Use max_length and truncation to adjust the text length
    tokenized = tokenizer(sample["text"], 
                          return_offsets_mapping=True, 
                          padding="max_length", 
                          max_length=MAX_LENGTH,
                          truncation=True)
    
    if 'offset_mapping' not in tokenized:
        raise KeyError("offset_mapping not found in tokenized output")
    
    labels = [label2id["O"]] * len(tokenized["input_ids"])  # Initialize labels with "O" (outside) tag

    for i, (token_start, token_end) in enumerate(tokenized["offset_mapping"]):
        for span in sample["tags"]:
            role = get_token_role_in_span(token_start, token_end, span["start"], span["end"])
            if role == "B":
                labels[i] = label2id[f"B-{span['tag']}"]
            elif role == "I":
                labels[i] = label2id[f"I-{span['tag']}"]

    tokenized.pop("offset_mapping")

    # Debugging statements
    print(f"Sample text: {sample['text']}")
    print(f"Tokenized text: {tokenizer.convert_ids_to_tokens(tokenized['input_ids'])}")
    print(f"Tags: {sample['tags']}")
    print(f"Labels: {labels}")
    
    return {**tokenized, "labels": labels}
