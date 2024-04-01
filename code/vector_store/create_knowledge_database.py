import gc
from pprint import pprint

import hydra
import pandas as pd
from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.milvus import MilvusVectorStore
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import StorageContext, ServiceContext
from llama_index.core.retrievers import QueryFusionRetriever
from langchain_community.retrievers import ElasticSearchBM25Retriever
from elasticsearch import Elasticsearch, helpers
from llama_index.embeddings.langchain import LangchainEmbedding
from omegaconf import DictConfig
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import Any, Iterable, List


class CustomElasticBM25Retriever(ElasticSearchBM25Retriever):
    n: int = 10
    """Number of hits to retrieve."""
    def _get_relevant_documents(
            self, query, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        query_dict = {"query": {"match": {"content": query}}, "size": self.n}
        res = self.client.search(index=self.index_name, body=query_dict)

        docs = []
        for r in res["hits"]["hits"]:
            docs.append(Document(page_content=r["_source"]["content"]))
        return docs


# Create a new vector store index
class KnowledgeDatabaseCreator:
    def __init__(self, config):
        self.data_config = config["dataset"]
        self.doc_config = config["document"]
        self.milvus_config = config["milvus"]
        self.MiniLM_config = config["model"]["MiniLM-L6-v2"]
        self.BGE_config = config["model"]["Bge-Base-En"]
        self.es_config = config["elasticsearch"]

    def load_documents(self):
        # Load documents from a directory
        print("Loading documents from", self.data_config["data_dir"])
        reader = SimpleDirectoryReader(
            input_dir=self.data_config["data_dir"],
            required_exts=self.data_config["file_postfixes"],
            recursive=True,
        )
        # Iterate over the documents
        for docs in reader.iter_data():
            for doc in docs:
                yield doc

    def create_elasticsearch_index(self):
        # Create a new ElasticSearch index
        es = Elasticsearch([self.es_config["url"]])
        es.indices.delete(index=self.es_config["index"], ignore_unavailable=True)
        es.indices.create(index=self.es_config["index"], ignore=400)

        client_info = es.info()
        print('Connected to Elasticsearch!')
        pprint(client_info.body)

        actions = []
        documents = self.load_documents()
        for doc in documents:
            doc_id = doc.get_doc_id()
            doc = {"id": doc_id, "content": doc.get_text()}
            action = {
                "_index": self.es_config["index"],
                "_id": doc_id,
                "_source": doc
            }
            actions.append(action)

            if len(actions) == self.es_config["bulk_size"]:
                helpers.bulk(es, actions)
                actions = []

        if actions:
            helpers.bulk(es, actions)

        es.indices.refresh(index=self.es_config["index"])
        es.cat.count(index=self.es_config["index"], format="json")

        del documents
        gc.collect()
        print("ElasticSearch index created")

    def create_milvus_index(self, embed_config, collection_name="collection"):
        # Create a Milvus vector store
        vector_store = MilvusVectorStore(
            url=self.milvus_config["url"],
            token=self.milvus_config["token"],
            collection_name=collection_name,
            dim=self.milvus_config["dim"],
            overwrite=True,
            index_config=self.milvus_config["index_config"]#add hnsw and inner product config
        )
        print("Milvus vector store created")

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        hf_embed_model = HuggingFaceEmbeddings(
            model_name=embed_config["model_path"]
        )
        embed_model = LangchainEmbedding(hf_embed_model)
        service_context = ServiceContext.from_defaults(embed_model=embed_model)

        # Create a vector store index
        index = VectorStoreIndex(
            storage_context=storage_context,
            service_context=service_context,
            chunk_size=self.doc_config["chunk_size"]
        )
        return index

    def create_milvus_db(self):
        # Create new vector store indexes using different embeddings
        MiniLM_index = self.create_milvus_index(self.MiniLM_config, collection_name="MiniLM")
        BGE_index = self.create_milvus_index(self.BGE_config, collection_name="Bge")

        # Update the indexes with the documents
        documents = self.load_documents()
        for doc in documents:
            MiniLM_index.update(doc, overwrite=False)
            BGE_index.update(doc, overwrite=False)

        del documents
        gc.collect()
        print("Milvus indexes created")

    def load_dense_retriever(self):
        # Load the milvus retriever
        # MiniLM_index = self.create_milvus_index(self.MiniLM_config)
        MiniLM_index= self.create_milvus_index(self.MiniLM_config, collection_name="MiniLM") #add collection name
        # BGE_index = self.create_milvus_index(self.BGE_config)
        BGE_index = self.create_milvus_index(self.BGE_config, collection_name="Bge")    #add collection name
        # Create a query fusion retriever
        dense_retriever = QueryFusionRetriever(
            [MiniLM_index.as_retriever(), BGE_index.as_retriever()],
            similarity_top_k=2*self.doc_config["top_k"],
            num_queries=1,  # set this to 1 to disable query generation
            use_async=True,
            verbose=True
        )
        return dense_retriever

    def load_sparse_retriever(self):
        # Load the elastic search retriever
        sparse_retriever = CustomElasticBM25Retriever.create(
            elasticsearch_url=self.es_config["url"],
            index_name=self.es_config["index"],
        )
        sparse_retriever.n = self.doc_config["top_k"]
        return sparse_retriever


@hydra.main(version_base=None, config_path="../../config", config_name="conf_database")
def main(cfg: DictConfig):
    creator = KnowledgeDatabaseCreator(cfg)

    if cfg["create_database"]:
        creator.create_elasticsearch_index()
        creator.create_milvus_db()

    # Load the retrievers
    dense_retriever = creator.load_dense_retriever()
    sparse_retriever = creator.load_sparse_retriever()

    query_df = pd.read_csv(cfg["dataset"]["test_query_file"])
    query_df["query"] = query_df[["prompt", "A", "B", "C", "D", "E"]].apply(lambda x: " | ".join(x), axis=1)

    for query in query_df["query"]:
        print("Query:", query)

        nodes_with_scores = dense_retriever.retrieve(query)
        print("Dense retriever results")
        for node in nodes_with_scores:
            print(f"Score: {node.score:.2f}")
            print(f"Content: {node.text[:100]}...")

        documents = sparse_retriever.get_relevant_documents(query)
        print("Sparse retriever results")
        for doc in documents:
            print(f"Content: {doc['page_content'][:100]}...")


if __name__ == "__main__":
    main()