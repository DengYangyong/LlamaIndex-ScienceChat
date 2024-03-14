import json
import os
import random
import time
from copy import deepcopy
from itertools import chain
import hydra
import pandas as pd
import torch
import torch.nn as nn
import wandb
from omegaconf import OmegaConf
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import List, Union, Any, Optional,Dict
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from dataclasses import dataclass
from omegaconf import DictConfig
import bitsandbytes as bnb
from sentence_transformers import InputExample
from llama_index.finetuning.cross_encoders.cross_encoder import (
    CrossEncoderFinetuneEngine,
)

@dataclass
class CrossEncoderFinetuningDatasetSample:
    """Class for keeping track of each item of Cross-Encoder training Dataset."""

    query: str
    context: str
    score: int
        
        
class RerankerCrossEncoder(CrossEncoder):
    def __init__(
        self,
        model_name: str,
        num_labels: int = None,
        max_length: int = None,
        device: str = None,
        tokenizer_args: Dict = {},
        automodel_args: Dict = {},
        revision: Optional[str] = None,
        default_activation_function=None,
        classifier_dropout: float = None,
    ):
        super().__init__(
            model_name=model_name,
            num_labels=num_labels,
            max_length=max_length,
            device=device,
            tokenizer_args=tokenizer_args,
            automodel_args=automodel_args,
            revision=revision,
            default_activation_function=default_activation_function,
            classifier_dropout=classifier_dropout,
        )

    def smart_batching_collate(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())

            labels.append(example.label)

        tokenized = self.tokenizer(
            *texts, padding=True, truncation="longest_first", return_tensors="pt", max_length=self.max_length, pad_to_multiple_of=64)
        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(
            self._target_device
        )

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized, labels

    
    
class RerankerFinetuneEngine(CrossEncoderFinetuneEngine):
    def __init__(
        self,
        dataset: List[CrossEncoderFinetuningDatasetSample],
        model_id: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        model_output_path: str = "exp_finetune",
        batch_size: int = 10,
        val_dataset: Union[List[CrossEncoderFinetuningDatasetSample], None] = None,
        loss: Union[Any, None] = None,
        epochs: int = 2,
        show_progress_bar: bool = True,
        evaluation_steps: int = 1000,
        max_length=1024,
        max_grad_norm=1,
        scheduler=None,
        weight_decay=0.01,
        beta1=0, 
        beta2=0,
        eps=0,
        lr=2e-5,
        use_bnb=False
        
    ) -> None:
        super().__init__(
            dataset=dataset,
            model_id=model_id,
            model_output_path=model_output_path,
            batch_size=batch_size,
            val_dataset=val_dataset,
            loss=loss,
            epochs=epochs,
            show_progress_bar=show_progress_bar,
            evaluation_steps=evaluation_steps,
        )
        self.model = RerankerCrossEncoder(self.model_id, num_labels=1, max_length=max_length)
        self.max_grad_norm=max_grad_norm
        self.scheduler=scheduler
        self.weight_decay=weight_decay
        self.lr = lr
        self.beta1=beta1
        self.beta2=beta2
        self.eps=eps
        self.use_bnb=use_bnb

    def finetune(self, **train_kwargs: Any) -> None:
        
        optimizer_params = {
            "betas": (self.beta1, self.beta2),
            "eps": self.eps,
            "lr": self.lr
        }
        
        if self.use_bnb:
            self.optimizer_class=bnb.optim.Adam8bit
        else:
            self.optimizer_class=torch.optim.AdamW
            
        self.model.fit(
            train_dataloader=self.loader,
            epochs=self.epochs,
            warmup_steps=self.warmup_steps,
            output_path=self.model_output_path,
            show_progress_bar=self.show_progress_bar,
            evaluator=self.evaluator,
            evaluation_steps=self.evaluation_steps,
            max_grad_norm=self.max_grad_norm,
            weight_decay=self.weight_decay,
            optimizer_params={"lr": self.lr},
            scheduler=self.scheduler if self.scheduler else "WarmupLinear",
            optimizer_class=self.optimizer_class)
    
        if self.evaluator is None:
            self.model.save(self.model_output_path)
        else:
            pass
 

class RerankerTrainer:
    def __init__(self, config):
        self.model_config = config["model"]
        self.train_config = config["train"]
        self.eval_config = config["eval"]
        self.debug = config["debug"]

    def load_dataset(self):
        train_dataset = pd.read_csv(self.train_config["train_data_file"])
        val_dataset = pd.read_csv(self.eval_config["val_data_file"])
        
        if self.debug:
            train_dataset = train_dataset.sample(n=200, random_state=42)

        train_finetuning_data_list = list(train_dataset.apply(lambda x: CrossEncoderFinetuningDatasetSample(query=x['query'], 
                                                             context=x['context'],
                                                             score=x['score']), axis=1))
        valid_finetuning_data_list = list(val_dataset.apply(lambda x: CrossEncoderFinetuningDatasetSample(query=x['query'], 
                                                             context=x['context'],
                                                             score=x['score']), axis=1))
        return train_finetuning_data_list, valid_finetuning_data_list

    def run_train(self,
                  train_finetuning_data_list: List[CrossEncoderFinetuningDatasetSample],
                  valid_finetuning_data_list: List[CrossEncoderFinetuningDatasetSample]):
        
        finetuning_engine = RerankerFinetuneEngine(
            model_id=self.model_config["model_name"],
            model_output_path=self.model_config["model_output_path"],
            dataset=train_finetuning_data_list, 
            val_dataset=valid_finetuning_data_list,
            evaluation_steps=self.train_config['evaluation_steps'],
            epochs=self.train_config["epochs"], 
            batch_size=self.train_config["batch_size"],
            lr=self.train_config["lr"],
            weight_decay=self.train_config["weight_decay"],
            beta1=self.train_config['beta1'],
            beta2=self.train_config['beta2'],
            eps=self.train_config['eps'],
            use_bnb=self.train_config['use_bnb']
        )

        # Finetune the cross-encoder model
        finetuning_engine.finetune()
    
    def run_evaluate(self, val_dataset: List[CrossEncoderFinetuningDatasetSample]):
        dev_samples = []
        for val_sample in val_dataset:
            val_query = val_sample.query
            val_text = val_sample.context
            val_score = val_sample.score
            val_example = InputExample(texts=[val_query, val_text], label=val_score)
            dev_samples.append(val_example)

        model = RerankerCrossEncoder(self.model_config['model_output_path'])
        evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_samples,  name="deberta")
        evaluator(model, output_path=self.eval_config['output_dir'])
        print("Evaluation done")
        
        
@hydra.main(version_base=None, config_path="../../config", config_name="conf_reranker")
def main(cfg: DictConfig):

    trainer = RerankerTrainer(cfg)
    train_dataset, val_dataset = trainer.load_dataset()
    print('Start Training')
    print('--------------------------------------------------')
    trainer.run_train(train_dataset, val_dataset)
    
    print('Start Evaluation')
    print('--------------------------------------------------')
    trainer.run_evaluate(val_dataset)
    
    
if __name__ == "__main__":
    main()