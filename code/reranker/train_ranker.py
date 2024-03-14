from pathlib import Path

import hydra
import pandas as pd
import torch

from typing import List, Any
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from omegaconf import DictConfig
import bitsandbytes as bnb
from sentence_transformers import InputExample
from llama_index.finetuning.cross_encoders.cross_encoder import CrossEncoderFinetuneEngine
from llama_index.finetuning.cross_encoders.dataset_gen import CrossEncoderFinetuningDatasetSample


class RerankerCrossEncoder(CrossEncoder):
    def smart_batching_collate(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())

            labels.append(example.label)

        # Add new parameter to use pad to multiple of 64
        tokenized = self.tokenizer(
            *texts, padding=True, truncation="longest_first", return_tensors="pt", max_length=self.max_length,
            pad_to_multiple_of=64)
        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(
            self._target_device
        )

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized, labels


class RerankerFinetuneEngine(CrossEncoderFinetuneEngine):
    def __init__(
        self, max_length: int = 512, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        # Set the max_length
        self.model = RerankerCrossEncoder(self.model_id, num_labels=1, max_length=max_length)

    def finetune(self, **train_kwargs: Any) -> None:

        optimizer_params = {
            "betas": (train_kwargs["beta1"], train_kwargs["beta2"]),
            "eps": train_kwargs["eps"],
            "lr": train_kwargs["lr"],
        }

        if train_kwargs["use_bnb"]:
            optimizer_class = bnb.optim.Adam8bit
        else:
            optimizer_class = torch.optim.AdamW

        for key in ["beta1", "beta2", "eps", "lr", "use_bnb"]:
            if key in train_kwargs:
                del train_kwargs[key]

        self.model.fit(
            train_dataloader=self.loader,
            epochs=self.epochs,
            warmup_steps=self.warmup_steps,
            output_path=self.model_output_path,
            show_progress_bar=self.show_progress_bar,
            evaluator=self.evaluator,
            evaluation_steps=self.evaluation_steps,
            optimizer_params=optimizer_params,
            optimizer_class=optimizer_class,
            **train_kwargs
        )

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
            val_dataset = val_dataset.sample(n=20, random_state=42)

        train_data_list = list(
            train_dataset.apply(lambda x: CrossEncoderFinetuningDatasetSample(query=x['query'],
                                                                              context=x['context'],
                                                                              score=x['score']), axis=1))
        valid_data_list = list(
            val_dataset.apply(lambda x: CrossEncoderFinetuningDatasetSample(query=x['query'],
                                                                            context=x['context'],
                                                                            score=x['score']), axis=1))
        return train_data_list, valid_data_list

    def run_train(self,
                  train_data_list: List[CrossEncoderFinetuningDatasetSample],
                  valid_data_list: List[CrossEncoderFinetuningDatasetSample]):

        finetune_engine = RerankerFinetuneEngine(
            model_id=self.model_config["model_name"],
            max_length=self.model_config["max_length"],
            model_output_path=self.model_config["model_output_path"],
            dataset=train_data_list,
            val_dataset=valid_data_list,
            evaluation_steps=self.train_config['evaluation_steps'],
            epochs=self.train_config["epochs"],
            batch_size=self.train_config["batch_size"]
        )

        # Finetune the cross-encoder model
        finetune_params = dict(
            lr=self.train_config["lr"],
            weight_decay=self.train_config["weight_decay"],
            beta1=self.train_config['beta1'],
            beta2=self.train_config['beta2'],
            eps=self.train_config['eps'],
            use_bnb=self.train_config['use_bnb']
        )
        finetune_engine.finetune(**finetune_params)

    def run_evaluate(self, val_dataset: List[CrossEncoderFinetuningDatasetSample]):
        dev_samples = [InputExample(
            texts=[sample.query, sample.context],
            label=sample.score
        ) for sample in val_dataset]

        model = RerankerCrossEncoder(self.model_config['model_output_path'])
        evaluator = CEBinaryClassificationEvaluator.from_input_examples(
            dev_samples, name="deberta"
        )
        Path(self.eval_config["output_dir"]).mkdir(exist_ok=True, parents=True)
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
