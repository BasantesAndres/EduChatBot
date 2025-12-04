# src/educhat/api.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .graph import compile_graph

app = FastAPI(title="EduChatAgent API", version="1.0")

# Permitir peticiones desde cualquier origen (para abrir el HTML localmente)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compilamos el grafo una sola vez
graph = compile_graph()


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Endpoint principal de chat.
    Recibe el mensaje del estudiante + session_id,
    ejecuta el grafo de LangGraph y devuelve la respuesta.
    """
    state = {"user_input": req.message}

    result = graph.invoke(
        state,
        config={"configurable": {"thread_id": req.session_id}},
    )

    answer = (
        result.get("final_answer")
        or result.get("json_answer")
        or "[No answer was generated]"
    )

    return ChatResponse(reply=answer)
