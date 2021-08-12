import re
import pandas as pd
import torch
from transformers import PreTrainedTokenizer


preprocess_pattern = re.compile(r"[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣]+")
repeat_pattern = re.compile(r"(.)\1{2,}")
space_pattern = re.compile(r"\s+")


def text_preprocessing(text):
    text = preprocess_pattern.sub("", text)
    text = repeat_pattern.sub(r"\1" * 3, text)
    text = space_pattern.sub(" ", text)
    return text.strip()


def df_to_feature_and_label(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    feature_col: str = "content",
    label_col: str = "gender_hate",
    max_length=128,
):
    tokens = tokenizer(
        list(df[feature_col]),
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    token_type_ids = tokens["token_type_ids"]
    labels = torch.from_numpy(df[label_col].values)

    return input_ids, attention_mask, token_type_ids, labels
