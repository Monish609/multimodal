from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def pdf_loader(file_path:str)-> list[Document]:
    """
        This is the Pdf Loader 

        #Input -->
            file_path  = file path of pdf
    """
    loader = PyPDFLoader(file_path=file_path,mode='page')
    docs = loader.load()

    return docs

def split_docs(docs:list[Document],chunk_size:int,chunk_overlap:int)-> list[Document]:
    """
        this is splitter which list of docs into chunk
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    chunk = splitter.split_documents(docs)

    return chunk
