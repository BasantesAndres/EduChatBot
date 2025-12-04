# src/educhat/cli.py

from datetime import datetime
import json
import os

from .graph import compile_graph

LOG_DIR = "logs/interactions"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def run_cli():
    print("🔄 Compiling EduChatAgent graph, please wait...\n")
    graph = compile_graph()
    ensure_dir(LOG_DIR)
    print("✅ EduChatAgent ready.\n")

    session = input("Session id (e.g. andres): ").strip() or "default"
    print("\nType your questions about the DATABASES course.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if user.lower() in {"exit", "quit", "salir"}:
            print("👋 Bye!")
            break

        # Estado inicial para el grafo
        state = {"user_input": user}
        print("\n[EduChatAgent] Generating answer, please wait...\n")

        # Invocamos LangGraph (usa RAG + chains + memoria)
        result = graph.invoke(
            state,
            config={"configurable": {"thread_id": session}},
        )

        answer = result.get("final_answer", "").strip()
        if not answer:
            answer = "[No answer generated]"

        print(f"\nEduChatAgent:\n{answer}\n")

        # Log a JSONL
        ensure_dir(LOG_DIR)
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session,
            "user_input": user,
            "state": result,
        }
        with open(
            os.path.join(LOG_DIR, f"{session}.jsonl"),
            "a",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_cli()
