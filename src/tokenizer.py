# from config import label2id, MAX_LENGTH, MODEL_PATH
# from transformers import RobertaTokenizerFast
# #from transformers import DistilBertTokenizerFast

# # Initialize the tokenizer
# tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
# #tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# def get_token_role_in_span(token_start: int, token_end: int, span_start: int, span_end: int):
#     """
#     Check if the token is inside a span.
#     Args:
#       - token_start, token_end: Start and end offset of the token
#       - span_start, span_end: Start and end of the span
#     Returns:
#       - "B" if beginning
#       - "I" if inner
#       - "O" if outer
#       - "N" if not valid token (like <SEP>, <CLS>, <UNK>)
#     """
#     if token_end <= token_start:
#         return "N"
#     if token_start < span_start or token_end > span_end:
#         return "O"
#     if token_start > span_start:
#         return "I"
#     else:
#         return "B"

# MAX_LENGTH = 50

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
    
#     # We are doing a multilabel classification task at each token, we create a list of size len(label2id)=13 
#     # for the 13 labels
#     labels = [[0 for _ in label2id.keys()] for _ in range(MAX_LENGTH)]
    
#     # Scan all the tokens and spans, assign 1 to the corresponding label if the token lies at the beginning
#     # or inside the spans
#     for (token_start, token_end), token_labels in zip(tokenized["offset_mapping"], labels):
#         for span in sample["tags"]:
#             role = get_token_role_in_span(token_start, token_end, span["start"], span["end"])
#             if role == "B":
#                 token_labels[label2id[f"B-{span['tag']}"]] = 1
#             elif role == "I":
#                 token_labels[label2id[f"I-{span['tag']}"]] = 1
    
#     return {**tokenized, "labels": labels}


# def test_tokenization_and_labels():
#     # Assume sample data
#     sample_data = {
#         "text": "Example text with entities like Microsoft located in Redmond.",
#         "tags": [{"start": 31, "end": 40, "tag": "ORG"}, {"start": 51, "end": 58, "tag": "LOC"}]
#     }
#     tokenized_sample = tokenize_and_adjust_labels(sample_data)
#     print(tokenized_sample)

# if __name__ == "__main__":
#     test_tokenization_and_labels()





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
    # Use max_length and truncation to ajust the text length
    tokenized = tokenizer(sample["text"], 
                          return_offsets_mapping=True, 
                          padding="max_length", 
                          max_length=MAX_LENGTH,
                          truncation=True)
    
    if 'offset_mapping' not in tokenized:
        raise KeyError("offset_mapping not found in tokenized output")
    
    labels = [[0 for _ in range(len(label2id) * 2)] for _ in range(len(tokenized["input_ids"]))]
    
    for i, (token_start, token_end) in enumerate(tokenized["offset_mapping"]):
        for span in sample["tags"]:
            role = get_token_role_in_span(token_start, token_end, span["start"], span["end"])
            if role == "B":
                labels[i][label2id[f"B-{span['tag']}"]] = 1
            elif role == "I":
                labels[i][label2id[f"I-{span['tag']}"]] = 1

    tokenized.pop("offset_mapping")

    # Debugging statements
    # print(f"Sample text: {sample['text']}")
    # print(f"Tokenized text: {tokenizer.convert_ids_to_tokens(tokenized['input_ids'])}")
    # print(f"Tags: {sample['tags']}")
    # print(f"Labels: {labels}")
    
    return {**tokenized, "labels": labels}