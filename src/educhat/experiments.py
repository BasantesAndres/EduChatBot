import json
from .graph import compile_graph
from .config import LLMConfig
from .llm_factory import make_hf_llm

# Aquí podrías cargar un pequeño conjunto de preguntas de prueba
TEST_QUESTIONS = [
    "Explain what HTTP is in simple terms.",
    "Give me 3 practice questions about the OSI model.",
    "When is the midterm exam?",
]

def run_with_config(config: LLMConfig, label: str):
    # Aquí podrías reconstruir el grafo usando este config
    # Para simplificar, supón que llm_factory usa config global;
    # en la práctica, puedes pasar config al construir el grafo.
    graph = compile_graph()

    results = []
    for q in TEST_QUESTIONS:
        state = {"user_input": q}
        out = graph.invoke(state, config={"configurable": {"thread_id": label}})
        results.append({"question": q, "final_answer": out.get("final_answer", "")})

    with open(f"logs/eval/results_{label}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def simple_eval():
    low = LLMConfig(temperature=0.1, top_p=0.85, top_k=40)
    high = LLMConfig(temperature=0.7, top_p=0.95, top_k=50)

    run_with_config(low, "low_temp")
    run_with_config(high, "high_temp")

    print("Saved results in logs/eval/. You can inspect them manually or in a notebook.")
