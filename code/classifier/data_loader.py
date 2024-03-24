import json
import os
from typing import Dict

import pandas as pd
import hydra
from omegaconf import DictConfig
from copy import deepcopy
import random

"""
<s>[INST] Use the provided input to create an instruction that could have been used to generate the response with an LLM.

{input} [/INST]

{response}</s>
"""


def create_prompt(sample):
    bos_token = "<s>"
    eos_token = "</s>"

    SYSTEM_PREFIX = """Below is an instruction that describes a task, paired with an input that provides further context. \
        Write a response that appropriately completes the request."""

    INSTRUCTION = """Your task is to analyze the question and answer below. Here A,B,C,D,E options are given choose the correct \
        one after Analyzing. As a potential aid to your answer, background context from Wikipedia articles is at your disposal, \
        even if they might not always be relevant."""

    question = sample['prompt']
    support = sample['support']  # retrieved context
    options = {
        "A": sample['A'],
        "B": sample['B'],
        "C": sample['C'],
        "D": sample['D'],
        "E": sample['E'],
    }
    response = "Answer: {}".format(sample['answer'])
    prompt_suffix = "".join([f"{letter}: {options[letter]}\n\n" for letter in "ABCDE"])
    input_prompt = f"Context: {support}\n\nQuestion: {question}\n\nOptions:\n{prompt_suffix}"

    full_prompt = ""
    full_prompt += bos_token
    full_prompt += "[INST]" + SYSTEM_PREFIX
    full_prompt += "\n" + INSTRUCTION
    full_prompt += "\n" + input_prompt
    full_prompt += "[/INST]"
    full_prompt += "\n" + response
    full_prompt += eos_token

    return full_prompt


def create_prompt_update(examples):
    bos_token = "<s>[INST] "  # 或者您模型的特定开始标记
    end_inst_token = " [/INST]"
    eos_token = "</s>"  # 或者您模型的特定结束标记

    SYSTEM_PREFIX = "Below is an instruction that describes a task, paired with an input that provides further context. \
        Write a response that appropriately completes the request.\n\n"

    INSTRUCTION = "Your task is to analyze the question and answer below. Here A,B,C,D,E options are given choose the correct \
        one after Analyzing. As a potential aid to your answer, background context from Wikipedia articles is at your disposal, \
        even if they might not always be relevant.\n\n"

    formatted_examples = []
    for prompt, support, A, B, C, D, E, answer in zip(
            examples['prompt'], examples['support'],
            examples['A'], examples['B'], examples['C'],
            examples['D'], examples['E'], examples['answer']):
        input_text = f"Backgrond Context to help you: \n {support}\n\n Question:\n {prompt}\n\n Options to choose from:\n A: {A}\n, B: {B}\n, C: {C}\n, D: {D}\n, E: {E}{end_inst_token}\n\n Answer:{answer}"

        full_text = f"{bos_token}{SYSTEM_PREFIX}{INSTRUCTION}{input_text}{eos_token}"

        formatted_examples.append(full_text)
    return formatted_examples


def sanitize_df_new(df):
    df = deepcopy(df)
    print(f"df shape before sanitize: {df.shape}")

    df = df[
        ~(df['prompt'].isna() | df['A'].isna() | df['B'].isna() | df['C'].isna() | df['D'].isna() | df['E'].isna())
    ].copy()

    df['valid_answer'] = df['answer'].apply(lambda x: x in ['A', 'B', 'C', 'D', 'E'])
    df = df[df['valid_answer']].copy()
    df = df.drop(columns=['valid_answer']).copy()
    df = df.reset_index(drop=True)

    df = df.reset_index(drop=True)
    print(f"df shape after sanitize: {df.shape}")

    return df


# 这个shuffle_answer_key函数的主要目的是在数据预处理过程中打乱（shuffle）每个问题的选项顺序
# ，同时保证答案（answer）标签的正确性不变。这是为了确保模型在学习时不会依赖于选项的固定顺序
def shuffle_answer_key(df):
    shuffled_df = deepcopy(df)
    # print_line()
    print(f"Answer Key Distribution Before Shuffling: {shuffled_df.answer.value_counts().sort_index()}")

    key2idx = {v: k for k, v in enumerate(list("ABCDE"))}
    idx2key = {v: k for k, v in key2idx.items()}

    shuffled_df["answer_string"] = shuffled_df[["A", "B", "C", "D", "E", "answer"]].apply(
        lambda x: x[key2idx[x[-1]]], axis=1
    )

    shuffled_df["options"] = shuffled_df[["A", "B", "C", "D", "E"]].apply(
        lambda x: random.sample(list(x), len(x)), axis=1
    )

    shuffled_df["A"] = shuffled_df["options"].apply(lambda x: x[0])
    shuffled_df["B"] = shuffled_df["options"].apply(lambda x: x[1])
    shuffled_df["C"] = shuffled_df["options"].apply(lambda x: x[2])
    shuffled_df["D"] = shuffled_df["options"].apply(lambda x: x[3])
    shuffled_df["E"] = shuffled_df["options"].apply(lambda x: x[4])

    shuffled_df["answer"] = shuffled_df[["A", "B", "C", "D", "E", "answer_string"]].apply(
        lambda x: idx2key[[idx for idx in range(5) if x[idx] == x[-1]][0]], axis=1
    )

    shuffled_df = shuffled_df[df.columns].copy()
    shuffled_df = shuffled_df.reset_index(drop=True)

    print(f"Answer Key Distribution After Shuffling: {shuffled_df.answer.value_counts().sort_index()}")
    return shuffled_df


def load_and_prepare_data(cfg: Dict):
    """
    Load training and validation datasets and their support information,
    then prepare them for fine-tuning a language model.

    Parameters:
    cfg (Dict): Configuration dictionary containing paths to datasets.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: Prepared training and validation DataFrames.
    """

    train_df = pd.read_csv(cfg['train_dataset_path'])
    print(f"shape of train data: {train_df.shape}")

    # Sanitize and shuffle (if these functions are defined elsewhere, make sure to import them)
    train_df = sanitize_df_new(train_df)
    train_df = shuffle_answer_key(train_df)

    # Load support information for training data
    with open(cfg['train_support_path'], 'r') as f:
        support_dict = json.load(f)
    train_df['support'] = train_df['id'].map(support_dict)
    assert train_df['support'].isna().sum() == 0, "Support is missing/invalid in training data."

    # ------- Load and prepare validation data -------------------------------------------#
    valid_df = pd.read_csv(cfg['valid_dataset_path'])
    valid_df = valid_df[['id', 'prompt', 'A', 'B', 'C', 'D', 'E', 'answer']].copy()
    valid_df['id'] = valid_df['id'].astype(str)

    # Load support information for validation data
    with open(cfg['valid_support_path'], 'r') as f:
        support_dict = json.load(f)
    valid_df['support'] = valid_df['id'].map(support_dict)
    assert valid_df['support'].isna().sum() == 0, "Support is missing/invalid in validation data."

    # print_line()
    train_df = train_df.rename(columns={"id": "question_id"})
    valid_df = valid_df.rename(columns={"id": "question_id"})

    return train_df, valid_df


@hydra.main(version_base=None, config_path="../../config", config_name="conf_llm")
def load_dataset(cfg: DictConfig):
    data_dir = cfg["dataset"]["data_dir"]

    data_cfg = {
        "train_dataset_path": os.path.join(data_dir, "train_mix_mcq.csv"),
        "train_support_path": os.path.join(data_dir, "id2context_k2_train.json"),
        "valid_dataset_path": os.path.join(data_dir, "valid_mix_mcq.csv"),
        "valid_support_path": os.path.join(data_dir, "id2context_k2_valid.json")
    }

    train_df, valid_df = load_and_prepare_data(data_cfg)

    if cfg["debug"]:
        train_df = train_df.sample(500)
        valid_df = valid_df.sample(100)

    print(f"shape of train data: {train_df.shape}")
    print(f"shape of valid data: {valid_df.shape}")

    # save to jsonl
    train_df.to_json(os.path.join(data_dir, 'train_mix_6.jsonl'), orient='records', lines=True)
    valid_df.to_json(os.path.join(data_dir, 'valid_mix_6.jsonl'), orient='records', lines=True)


if __name__ == "__main__":
    load_dataset()
