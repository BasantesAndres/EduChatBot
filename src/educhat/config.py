# src/educhat/config.py
from dataclasses import dataclass

@dataclass
class LLMConfig:
    model_id: str
    task: str = "chat"  # ya no usamos HF pipeline, pero lo dejamos por compatibilidad
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 40
    max_new_tokens: int = 128 # podemos usarlo más adelante si queremos limitar

# Configuración por defecto: Ollama con gemma3:4b
DEFAULT_CONFIG_LOW_TEMP = LLMConfig(
    model_id="gemma3:4b",  # nombre del modelo en ollama list
    temperature=0.2,
    top_p=0.9,
    top_k=40,
    max_new_tokens=128,
)
