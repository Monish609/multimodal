from langchain_community.document_loaders import PyPDFLoader

def pdf_loader(file_path:str)-> list[any]:
    """
        This is the Pdf Loader 

        #Input -->
            file_path  = file path of pdf
    """
    loader = PyPDFLoader(file_path=file_path,mode='page')
    docs = loader.load()

    return docs







