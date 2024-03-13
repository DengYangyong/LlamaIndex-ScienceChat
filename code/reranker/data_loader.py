import os
import pandas as pd
import hydra
from omegaconf import DictConfig

def pre_process(df):
    columns = ["prompt", "A", "B", "C", "D", "E"]
    df["query"] = df[columns].apply(lambda x: " | ".join(x), axis=1)
    return df

@hydra.main(version_base=None, config_path="../../config", config_name="conf_reranker")
def load_and_process_data(cfg: DictConfig):
    data_dir = cfg["dataset"]["data_dir"]
    data_df = pd.read_parquet(os.path.join(data_dir, "train_ranker_dataset.parquet"))
    data_df = data_df.rename(columns={"id": "query_id"})
    train_df = data_df[data_df["is_train"] == 0].copy()
    valid_df = data_df[data_df["is_train"] == 1].copy()

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    print(f"shape of train data: {train_df.shape}")
    print(f"shape of validation data: {valid_df.shape}")

    train_ds = pre_process(train_df)[['query', 'context', 'label']].rename(columns={'label': 'score'})
    valid_ds = pre_process(valid_df)[['query', 'context', 'label']].rename(columns={'label': 'score'})
    
    train_ds.to_csv(os.path.join(data_dir, "train_dataset.csv"))
    valid_ds.to_csv(os.path.join(data_dir, "val_dataset.csv"))

if __name__=="__main__":
    load_and_process_data()
    
