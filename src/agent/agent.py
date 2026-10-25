from dotenv import load_dotenv
import os

from langchain_ollama import ChatOllama
from model import get_model, get_response

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")


def improve_prompt(llm: ChatOllama, user_input: str) -> str:
    """
    Improve the user input prompt.

    Args:
        llm (ChatOllama): The language model instance.
        user_input (str): The original user input.

    Returns:
        str: The improved prompt.
    """
    prompt = f"""Rewrite the user's input as a high-quality AI prompt. 
    Preserve intent, add missing context when needed, and make it clear, specific, and self-contained. 
    If the input is vague, expand it into a useful request. Return only the rewritten prompt.

Input:
{user_input}"""
    
    response = get_response(llm, prompt)
    return response

def main():
    llm = get_model(OLLAMA_MODEL, OLLAMA_BASE_URL)
    user_input = "what is AI?"
    improved_prompt = improve_prompt(llm, user_input)
    response = get_response(llm, improved_prompt)
    print(f"Assistant: {response}")

