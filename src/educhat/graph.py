# src/educhat/graph.py

from typing import TypedDict, Literal, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from .config import DEFAULT_CONFIG_LOW_TEMP
from .llm_factory import make_hf_llm
from .chains import (
    build_memory,
    build_router_chain,
    build_faq_chain,
    build_concept_sequential_chain,
    build_practice_chain,
)
from .tools import course_rag_search


class EduChatState(TypedDict, total=False):
    user_input: str
    mode: Literal["faq", "concept", "practice"]
    history: str
    retrieved_context: Optional[str]
    draft_answer: Optional[str]
    json_answer: Optional[str]
    final_answer: str


def build_educhat_graph() -> StateGraph:
    # LLM con configuración por defecto (baja temperatura)
    llm = make_hf_llm(DEFAULT_CONFIG_LOW_TEMP)

    # Memoria tipo buffer (para conversaciones)
    memory = build_memory()

    # Chains de LangChain (clásicas)
    router_chain = build_router_chain(llm)
    faq_chain = build_faq_chain(llm, memory)
    concept_chain = build_concept_sequential_chain(llm, memory)
    practice_chain = build_practice_chain(llm)

    # Creamos el grafo de estado
    builder = StateGraph(EduChatState)

    # -------- NODOS -------- #

    def input_node(state: EduChatState) -> EduChatState:
        return state

    def router_node(state: EduChatState) -> EduChatState:
        # invoke devuelve un dict con la clave "mode"
        out = router_chain.invoke({"user_input": state["user_input"]})
        mode = out.get("mode", "").strip().lower()
        if mode not in {"faq", "concept", "practice"}:
            mode = "concept"  # fallback razonable
        state["mode"] = mode
        return state

    def faq_node(state: EduChatState) -> EduChatState:
        out = faq_chain.invoke({"user_input": state["user_input"]})
        answer = out.get("answer", "")
        state["final_answer"] = answer
        return state

    def concept_node(state: EduChatState) -> EduChatState:
        # 1) RAG: obtenemos contexto del curso
        context = course_rag_search(state["user_input"])

        # 2) Ejecutamos la SequentialChain con ese contexto
        out = concept_chain.invoke(
            {
                "user_input": state["user_input"],
                "history": state.get("history", ""),
                "retrieved_context": context,
            }
        )

        state["retrieved_context"] = context
        state["draft_answer"] = out.get("draft_answer")
        state["json_answer"] = out.get("json_answer")
        state["final_answer"] = out.get("json_answer", state.get("draft_answer", ""))

        return state

    def practice_node(state: EduChatState) -> EduChatState:
        # 1) RAG: contexto del curso para generar ejercicios relevantes
        context = course_rag_search(state["user_input"])

        # IMPORTANTE: structured_json_prompt espera también "draft_answer",
        # así que le pasamos una cadena vacía para satisfacer la firma.
        out = practice_chain.invoke(
            {
                "user_input": state["user_input"],
                "retrieved_context": context,
                "draft_answer": "",  # <- añadido
            }
        )

        state["retrieved_context"] = context
        state["json_answer"] = out.get("json_answer")
        state["final_answer"] = out.get("json_answer", "")
        return state


    def memory_node(state: EduChatState) -> EduChatState:
        prev = state.get("history", "")
        new_turn = f"User: {state['user_input']}\nAgent: {state.get('final_answer', '')}\n"
        state["history"] = (prev + "\n" + new_turn).strip()
        return state

    def final_node(state: EduChatState) -> EduChatState:
        return state

    # -------- REGISTRO DE NODOS -------- #

    builder.add_node("input", input_node)
    builder.add_node("router", router_node)
    builder.add_node("faq_node", faq_node)
    builder.add_node("concept_node", concept_node)
    builder.add_node("practice_node", practice_node)
    builder.add_node("memory_node", memory_node)
    builder.add_node("final_node", final_node)

    # -------- ARISTAS -------- #

    builder.add_edge(START, "input")
    builder.add_edge("input", "router")

    # Ruteo condicional según el modo
    def mode_selector(state: EduChatState) -> str:
        return state["mode"]

    builder.add_conditional_edges(
        "router",
        mode_selector,
        {
            "faq": "faq_node",
            "concept": "concept_node",
            "practice": "practice_node",
        },
    )

    builder.add_edge("faq_node", "memory_node")
    builder.add_edge("concept_node", "memory_node")
    builder.add_edge("practice_node", "memory_node")

    builder.add_edge("memory_node", "final_node")
    builder.add_edge("final_node", END)

    return builder


def compile_graph():
    """
    Compila el grafo de EduChatAgent con un checkpointer en memoria
    para poder tener sesiones (thread_id).
    """
    builder = build_educhat_graph()
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    return graph
