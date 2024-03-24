import sys
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from llama_index.core.postprocessor import SentenceTransformerRerank
from sentence_transformers.cross_encoder import CrossEncoder
from reranker.train_ranker import RerankerCrossEncoder
from vector_store.create_knowledge_database import KnowledgeDatabaseCreator
from classifier.train_llm import generate
from classifier.data_loader import create_prompt
import torch


class KnownLedgeBaseQA:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dense_retriever = None
        self.sparse_retriever = None
        self.reranker = None
        self.llm = None
        self.tokenizer = None

    def load_retriever(self):
        creator = KnowledgeDatabaseCreator(self.cfg["retriever"])
        self.dense_retriever = creator.load_dense_retriever()
        self.sparse_retriever = creator.load_sparse_retriever()

    def load_reranker(self):
        self.reranker = RerankerCrossEncoder(
            model=self.cfg["rerank"]["model_name_or_path"],
            max_length=self.cfg["rerank"]["max_length"],
            top_n=3
        )

    def load_llm(self):
        self.llm = AutoModelForCausalLM.from_pretrained(self.cfg["model"]["model_name_or_path"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["model"]["model_name_or_path"])

    def inference(self, sample):
        question = sample['prompt']

        # Retrieve documents
        retriever_results = []
        nodes_with_scores = self.dense_retriever.retrieve(question)
        for node in nodes_with_scores:
            retriever_results.append(node.text)

        documents = self.sparse_retriever.get_relevant_documents(question)
        for doc in documents:
            retriever_results.append(doc['page_content'])

        # Rerank documents
        reranker_results = self.reranker.rank(
            question,
            retriever_results,
            top_k=self.cfg["rerank"]["top_k"],
            num_workers=self.cfg["rerank"]["num_workers"],
            batch_size=self.cfg["rerank"]["batch_size"],
            return_documents=True
        )

        reranker_contents = [result['text'] for result in reranker_results]
        sample['support'] = " | ".join(reranker_contents)

        # Generate answer
        prompt = create_prompt(sample)
        answer = generate(
            prompt,
            self.tokenizer,
            self.llm,
            max_new_tokens=self.cfg["llm"]["max_new_tokens"]
        )
        return answer