from ingestion.datahandler import pdf_loader,split_docs
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document

def get_embedding_model():
    """
        Set Embedding model
    """
    return OllamaEmbeddings(
            model= "embeddinggemma-2:570m"
        )


def db_connection(url:str,collection_name:str) -> PGVector:
    """
        Connect to DB
    """
    embedding = get_embedding_model()

    vector_store = PGVector(
        connection=url,
        collection_name = collection_name,
        embeddings=embedding,
        use_jsonb=True,
    )

    return vector_store


def embed_document(docs:list[Document],vector_store:PGVector):
    """
        Embedding the documents 
    """
    doc_ids = vector_store.add_documents(docs)
    return doc_ids


def similarity_search(query:str,vector_store:PGVector)-> list[Document]:
    """
        Top K result from document
    """
    result = vector_store.similarity_search(query=query,k=5)

    return result