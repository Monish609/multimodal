from langchain_core.documents import Document
from langchain_community.vectorstores import PGVector

def similarity_search(query:str,vector_store:PGVector,k:int = 5)-> list[Document]:
    """
        Top K result from document
    """
    result = vector_store.similarity_search_with_score(query=query,k=k)

    return result
