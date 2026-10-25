from langchain_ollama import ChatOllama

def get_model(OLLAMA_MODEL: str, OLLAMA_BASE_URL: str) -> ChatOllama:
    """
    Get the Ollama model instance.

    Returns:
        ChatOllama: The Ollama model instance.
    """

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    return llm

def get_response(llm: ChatOllama, user_input: str) -> str:
    """
    Get a response from the Ollama model based on the provided user input.

    Args:
        llm (ChatOllama): The Ollama model instance.
        user_input (str): The input text to send to the Ollama model.

    Returns:
        str: The model's response text.
    """
    response = llm.invoke(user_input)
    return response.content
