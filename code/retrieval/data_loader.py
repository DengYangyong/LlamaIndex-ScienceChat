import json
import os
import pandas as pd
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../config", config_name="conf_retriever")
def load_dataset(cfg: DictConfig):
    data_dir = cfg["dataset"]["data_dir"]
    # load dataframe
    query_df = pd.read_parquet(os.path.join(data_dir, "sci_queries.parquet"))
    content_df = pd.read_parquet(os.path.join(data_dir, "sci_contents.parquet"))
    labels_df = pd.read_parquet(os.path.join(data_dir, "sci_labels.parquet"))
    fold_df = pd.read_parquet(os.path.join(data_dir, "sci_folds.parquet"))

    if cfg["debug"]:
        fold_df = fold_df.sample(n=200, random_state=42)

    # split data into train, val
    is_train = fold_df['is_train'] == 1

    labels_df_train = labels_df[labels_df["query_id"].isin(fold_df[is_train]["query_id"])]
    labels_df_val = labels_df[labels_df["query_id"].isin(fold_df[~is_train]["query_id"])]
    print("Train size: ", labels_df_train.shape[0])
    print("Val size: ", labels_df_val.shape[0])

    query_df_train = query_df[query_df["query_id"].isin(labels_df_train["query_id"])]
    query_df_val = query_df[query_df["query_id"].isin(labels_df_val["query_id"])]

    content_df_train = content_df[content_df["content_id"].isin(labels_df_train["content_id"])]
    content_df_val = content_df[content_df["content_id"].isin(labels_df_val["content_id"])]

    # convert data to json
    train_json = data_to_json(query_df_train, content_df_train, labels_df_train)
    val_json = data_to_json(query_df_val, content_df_val, labels_df_val)

    # save json
    with open(os.path.join(data_dir, "train_dataset.json"), "w") as f:
        json.dump(train_json, f)

    with open(os.path.join(data_dir, "val_dataset.json"), "w") as f:
        json.dump(val_json, f)


def data_to_json(query_df, content_df, labels_df):
    id2queries = dict(
        zip(query_df['query_id'], query_df[["prompt", "A", "B", "C", "D", "E"]].apply(lambda x: " | ".join(x), axis=1))
    )

    id2content = dict(
        zip(content_df['content_id'], content_df['context'])
    )

    # format: {"query_id": [content_id, ...]}
    relevant_docs = labels_df.groupby('query_id')['content_id'].apply(list).to_dict()

    json_data = {
        "queries": id2queries,
        "corpus": id2content,
        "relevant_docs": relevant_docs,
        "mode": "text"
    }

    return json_data


if __name__ == "__main__":
    load_dataset()
