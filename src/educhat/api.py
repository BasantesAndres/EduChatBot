# src/educhat/api.py

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .graph import compile_graph

app = FastAPI(title="EduChatAgent API")

graph = compile_graph()  # compila LangGraph una vez al iniciar

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok", "message": "EduChatAgent API running"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    state = {"user_input": req.message}
    result = graph.invoke(
        state,
        config={"configurable": {"thread_id": req.session_id}},
    )
    answer = result.get("final_answer", "")
    return ChatResponse(answer=answer)


# Interfaz web muy sencilla en la raíz "/"
@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>EduChatAgent</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 2rem auto; }
    #chat { border: 1px solid #ccc; padding: 1rem; height: 400px; overflow-y: auto; }
    .msg-user { font-weight: bold; margin-top: 0.5rem; }
    .msg-bot { margin-left: 1rem; margin-bottom: 0.5rem; }
    #input-row { margin-top: 1rem; display: flex; gap: 0.5rem; }
    #message { flex: 1; padding: 0.5rem; }
    button { padding: 0.5rem 1rem; cursor: pointer; }
  </style>
</head>
<body>
  <h1>EduChatAgent – Course Tutor</h1>
  <div id="chat"></div>
  <div id="input-row">
    <input id="message" type="text" placeholder="Ask something about the course..." />
    <button onclick="sendMessage()">Send</button>
  </div>

  <script>
    const sessionId = "andres-demo"; // podrías hacerlo aleatorio si quieres

    async function sendMessage() {
      const input = document.getElementById("message");
      const text = input.value.trim();
      if (!text) return;

      const chat = document.getElementById("chat");
      chat.innerHTML += `<div class="msg-user">You: ${text}</div>`;

      input.value = "";
      input.focus();

      const resp = await fetch("/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ session_id: sessionId, message: text })
      });

      const data = await resp.json();
      const answer = data.answer || "[no answer]";
      chat.innerHTML += `<div class="msg-bot">EduChatAgent: ${answer}</div>`;
      chat.scrollTop = chat.scrollHeight;
    }

    // Enter = enviar
    document.getElementById("message").addEventListener("keydown", function(e) {
      if (e.key === "Enter") {
        sendMessage();
      }
    });
  </script>
</body>
</html>
"""
