import os
from typing import Any

import hydra
from llama_index.finetuning import EmbeddingQAFinetuneDataset, SentenceTransformersFinetuneEngine
from omegaconf import DictConfig
from sentence_transformers.losses import MultipleNegativesSymmetricRankingLoss
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from tqdm.notebook import tqdm
import pandas as pd
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from pathlib import Path
from sentence_transformers import SentenceTransformer
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.embeddings import HuggingFaceEmbeddings


class RetrieverFinetuneEngine(SentenceTransformersFinetuneEngine):
    def finetune(self, **train_kwargs: Any) -> None:
        """Finetune model."""
        self.model.fit(
            train_objectives=[(self.loader, self.loss)],
            epochs=self.epochs,
            warmup_steps=self.warmup_steps,
            output_path=self.model_output_path,
            show_progress_bar=self.show_progress_bar,
            evaluator=self.evaluator,
            evaluation_steps=self.evaluation_steps,
            **train_kwargs
        )


class RetrieverTrainer:
    def __init__(self, config):
        self.model_config = config["model"]
        self.train_config = config["train"]
        self.eval_config = config["eval"]
        self.embed_model = None

    def load_dataset(self):
        train_dataset = EmbeddingQAFinetuneDataset.from_json(self.train_config["train_data_file"])
        val_dataset = EmbeddingQAFinetuneDataset.from_json(self.eval_config["val_data_file"])
        return train_dataset, val_dataset

    def run_train(self,
                  train_dataset: EmbeddingQAFinetuneDataset,
                  val_dataset: EmbeddingQAFinetuneDataset):

        # define loss
        model = SentenceTransformer(self.model_config["model_name"])
        loss = MultipleNegativesSymmetricRankingLoss(model=model)

        finetune_engine = RetrieverFinetuneEngine(
            train_dataset,
            model_id=self.model_config["model_name"],
            model_output_path=self.model_config["model_output_path"],
            batch_size=self.train_config["batch_size"],
            val_dataset=val_dataset,
            loss=loss,
            epochs=self.train_config["epochs"],
            evaluation_steps=self.train_config["evaluation_steps"],
            use_all_docs=self.train_config["use_all_docs"],
        )

        # use mixed precision
        finetune_engine.finetune(use_amp=self.train_config["use_mixed_precision"])
        self.embed_model = finetune_engine.get_finetuned_model()

    def run_evaluate(self, val_dataset: EmbeddingQAFinetuneDataset):

        _ = self.ir_evaluator(val_dataset)
        model_name = self.model_config["model_name"].split("/")[-1]
        eval_result_file = "Information-Retrieval_evaluation_" + model_name + "_results.csv"
        df_eval_metrics = pd.read_csv(os.path.join(self.eval_config["output_dir"], eval_result_file))
        print("Evaluation metrics: ")
        for metric, value in df_eval_metrics.mean().items():
            print(f"{metric}: {value}")

        hit_rate = self.hit_rate_evaluator(val_dataset)
        print("Hit rate: ", hit_rate)

        print("Evaluation done")

    def hit_rate_evaluator(self, val_dataset: EmbeddingQAFinetuneDataset):
        corpus = val_dataset.corpus
        queries = val_dataset.queries
        relevant_docs = val_dataset.relevant_docs

        if self.embed_model is None:
            hf_embed_model = HuggingFaceEmbeddings(
                model_name=self.model_config["model_output_path"]
            )
            self.embed_model = LangchainEmbedding(hf_embed_model)

        nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
        index = VectorStoreIndex(
            nodes, embed_model=self.embed_model, show_progress=True
        )
        retriever = index.as_retriever(similarity_top_k=self.eval_config["top_k"])

        eval_results = []
        for query_id, query in tqdm(queries.items()):
            retrieved_nodes = retriever.retrieve(query)
            retrieved_ids = [node.node.node_id for node in retrieved_nodes]
            expected_id = relevant_docs[query_id][0]  # assume 1 relevant doc
            is_hit = expected_id in retrieved_ids

            eval_result = {
                "is_hit": is_hit,
                "retrieved": retrieved_ids,
                "expected": expected_id,
                "query": query_id,
            }
            eval_results.append(eval_result)

        df_val = pd.DataFrame(eval_results)
        hit_rate_ada = df_val["is_hit"].mean()
        return hit_rate_ada

    def ir_evaluator(self, val_dataset: EmbeddingQAFinetuneDataset):
        corpus = val_dataset.corpus
        queries = val_dataset.queries
        relevant_docs = val_dataset.relevant_docs
        relevant_docs = {query_id: set(doc_ids) for query_id, doc_ids in relevant_docs.items()}

        # Load the finetuned model
        embed_model = SentenceTransformer(self.model_config["model_output_path"])
        model_name = self.model_config["model_name"].split("/")[-1]
        evaluator = InformationRetrievalEvaluator(
            queries, corpus, relevant_docs, name=model_name, show_progress_bar=True
        )

        Path(self.eval_config["output_dir"]).mkdir(exist_ok=True, parents=True)
        return evaluator(embed_model, output_path=self.eval_config["output_dir"])


# 自动化配置
@hydra.main(version_base=None, config_path="../../config", config_name="conf_retriever")
def main(cfg: DictConfig):

    trainer = RetrieverTrainer(cfg)
    train_dataset, val_dataset = trainer.load_dataset()
    trainer.run_train(train_dataset, val_dataset)
    trainer.run_evaluate(val_dataset)


if __name__ == "__main__":
    main()
