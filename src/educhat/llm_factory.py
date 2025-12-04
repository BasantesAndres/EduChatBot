# src/educhat/llm_factory.py

"""
Factoría de LLMs para EduChat.

Antes usábamos un modelo de HuggingFace en local (transformers + pipeline).
Ahora usamos un modelo de Ollama (gemma3:4b) vía ChatOllama, que es más ligero
para tu máquina y cumple con la rúbrica de usar LLMs open-source (Ollama).
"""

from langchain_ollama import ChatOllama  # integración oficial Ollama + LangChain
from .config import LLMConfig


def make_hf_llm(config: LLMConfig):
    """
    Crea un LLM de Ollama usando la configuración dada.

    Mantenemos el nombre `make_hf_llm` para no cambiar nada en graph.py,
    pero internamente ya no usa HuggingFace, sino ChatOllama.
    """
    llm = ChatOllama(
        model=config.model_id,           # ej. "gemma3:4b"
        temperature=config.temperature,  # control de aleatoriedad
        top_p=config.top_p,
        # num_predict=config.max_new_tokens,  # si quieres limitar tokens de salida
        # puedes descomentar esta línea
    )
    return llm
