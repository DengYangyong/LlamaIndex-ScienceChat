from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.milvus import MilvusVectorStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import StorageContext, ServiceContext
from llama_index.core.retrievers import QueryFusionRetriever


# Create a new vector store index
class CreateVectorStore:
    def __init__(self, config):
        self.chunk_size = config["chunk_size"]
        self.milvus_config = config["milvus"]
        self.MiniLM_config = config["embeddings"]["MiniLM-L6-v2"]
        self.BGE_config = config["embeddings"]["Bge-Base-En"]
        self.es_config = config["elasticsearch"]
        self.top_k = config["top_k"]
        self.dense_retriever = None
        self.sparse_retriever = None


    @staticmethod
    def load_documents(data_dir, file_postfixes):
        # Load documents from a directory
        reader = SimpleDirectoryReader(
            input_dir=data_dir,
            required_exts=file_postfixes,
            recursive=True,
        )

        # Iterate over the documents
        for docs in reader.iter_data():
            for doc in docs:
                yield doc

    def create_elasticsearch_index(self):
        pass

    def create_milvus_index(self, embed_config):
        # Create a Milvus vector store
        vector_store = MilvusVectorStore(
            url=self.milvus_config["url"],
            token=self.milvus_config["token"],
            collection_name=self.milvus_config["collection_name"],
            dim=self.milvus_config["dim"],
            overwrite=True
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        embed_model = HuggingFaceEmbeddings(model_name=embed_config["model_name"],
                                            cache_folder=embed_config["model_path"]
        )
        service_context = ServiceContext.from_defaults(embed_model=embed_model)

        # Create a vector store index
        index = VectorStoreIndex(storage_context=storage_context,
                                 service_context=service_context,
                                 chunk_size=self.chunk_size
        )
        return index

    def create_milvus_db(self, data_dir, file_postfixes):
        # Create new vector store indexes using different embeddings
        MiniLM_index = self.create_milvus_index(self.MiniLM_config)
        BGE_index = self.create_milvus_index(self.BGE_config)

        documents = self.load_documents(data_dir, file_postfixes)
        for doc in documents:
            MiniLM_index.update(doc, overwrite=False)
            BGE_index.update(doc, overwrite=False)

    def load_dense_retriever(self):
        # Load the milvus retriever
        MiniLM_index = self.create_milvus_index(self.MiniLM_config)
        BGE_index = self.create_milvus_index(self.BGE_config)

        # Create a query expansion retriever
        QUERY_GEN_PROMPT = (
            "You are a helpful assistant that generates multiple search queries based on a "
            "single input query. Generate {num_queries} search queries, one on each line, "
            "related to the following input query:\n"
            "Query: {query}\n"
            "Queries:\n"
        )

        # Create a query fusion retriever
        self.dense_retriever = QueryFusionRetriever(
            [MiniLM_index.as_retriever(), BGE_index.as_retriever()],
            similarity_top_k=2*self.top_k,
            num_queries=1,  # set this to 1 to disable query generation
            use_async=True,
            verbose=True,
            # query_gen_prompt=QUERY_GEN_PROMPT,  # we could override the query generation prompt here
        )