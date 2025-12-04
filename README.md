# 📚 EduChatAgent – Databases Course Intelligent Tutor

**Author:** Andrés Basantes  
**Course:** Intelligent Agents – Yachay Tech  
**Target Course:** Databases (UC1)

---

## 🧠 Project Overview

EduChatAgent is an **LLM-powered educational assistant** specialized in the **Databases** course.  
It answers questions about:

- ✅ **Course logistics** – schedule, evaluation, final project weight  
- ✅ **Theory & concepts** – relational model, SQL, APIs, NoSQL  
- ✅ **Practice** – quiz-style questions and exercises based on course material  
- ✅ **Bibliography** – main books and reference material from the syllabus  

The agent is built using:

- 🧩 **LangChain** – prompt templates, chains, memory  
- 🔀 **LangGraph** – graph-based workflow (router + tools + memory)  
- 🧠 **Ollama** – open LLM `gemma3:4b` for local inference  
- 🧮 **RAG** – Retrieval-Augmented Generation over syllabus, units, quizzes, evaluation and bibliography  
- 🧱 **FastAPI** – REST API  
- 💻 **Minimalist Web UI** – HTML/CSS/JS single-page chat interface  

---

## 🎯 Objectives

1. **Design and implement a full LLM agent pipeline**:
   - Environment ↔ LLM Reasoning ↔ Tools/Actions ↔ Output.
2. **Demonstrate LangChain competencies**:
   - Prompt templates, router chain, memory, (sequential) chains.
3. **Demonstrate LangGraph competencies**:
   - Graph with router node, multiple nodes, memory and tools.
4. **Use open-source LLMs**:
   - Local inference with **Ollama `gemma3:4b`**.
5. **Integrate RAG for course grounding**:
   - Syllabus, UC1–UC4 contents, quizzes, evaluation scheme, bibliography.
6. **Expose the agent via API + Web UI** for an interactive educational experience.

---

## 🏗️ High-Level Architecture

```text
           ┌─────────────────────┐
           │  Raw Course Docs    │
           │  (txt syllabus,     │
           │   UC1–UC4, quizzes, │
           │   evaluation, bibl.)│
           └─────────┬───────────┘
                     │  build_rag.py
                     ▼
             ┌────────────────┐
             │ Vector Store   │
             │ (Chroma +      │
             │  MiniLM emb.)  │
             └──────┬─────────┘
                    │  course_rag_search()
                    ▼
┌──────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                        │
│                                                              │
│  START → input → router → { faq | concept | practice }       │
│                         │          │            │            │
│                         └──────→ memory_node → final → END   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                    │
                    ▼
          ┌─────────────────────┐
          │  Ollama LLM         │
          │  gemma3:4b          │
          └─────────────────────┘
                    │
                    ▼
           CLI / FastAPI / Web UI
```

---

## 📂 Project Structure

```text
EduChatBot/
│
├── src/
│   ├── educhat/
│   │   ├── __init__.py
│   │   ├── config.py          # LLM configuration (Ollama gemma3:4b, params)
│   │   ├── llm_factory.py     # Builds ChatOllama LLM from config
│   │   ├── prompts.py         # All prompt templates (router, FAQ, concept, JSON)
│   │   ├── chains.py          # LangChain chains (router, FAQ, concept, practice, memory)
│   │   ├── tools.py           # RAG tool: course_rag_search()
│   │   ├── rag_store.py       # Build/load vector store (Chroma + MiniLM embeddings)
│   │   ├── build_rag.py       # CLI to index course docs in data/raw
│   │   ├── graph.py           # LangGraph workflow (nodes, state, routing)
│   │   ├── cli.py             # Terminal chatbot client
│   │   ├── api.py             # FastAPI app exposing POST /chat
│   │   └── ...
│   │
│   └── data/
│       ├── raw/               # ✨ Source text files for RAG
│       │   ├── syllabus.txt
│       │   ├── uc_contents.txt
│       │   ├── Contenido UC1.txt
│       │   ├── evaluation.txt
│       │   ├── bibliography.txt
│       │   └── quizzes_*.txt
│       └── processed/
│           └── chroma/        # Auto-generated Chroma DB (ignored by git)
│
├── web/
│   └── index.html             # Minimalist single-page chat UI
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧩 RAG: Retrieval-Augmented Generation

The agent is grounded on the actual content of the Databases course.

### 📁 Data sources (in `src/data/raw/`)

- `syllabus.txt` – official course description, learning outcomes.
- `uc_contents.txt` – UC1–UC4 titles and topics:
  - UC1 – Fundamentals and Database Design  
  - UC2 – SQL (DDL, DML, joins, aggregation, functions)  
  - UC3 – APIs for database operations  
  - UC4 – NoSQL databases.
- `Contenido UC1.txt` – detailed UC1 theory and instructor notes.
- `evaluation.txt` – evaluation scheme (percentages, final project weight).
- `bibliography.txt` – main books and references.
- `quizzes_*.txt` – past quiz questions to inspire practice questions.

### 🔧 Building the vector store

RAG is built with:

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector store:** ChromaDB

```bash
# From project root
cd src
python -m educhat.build_rag
```

You should see something like:

```text
Loaded N documents from data/raw
✅ Vector store built in data/processed/chroma
```

### 🧰 RAG tool – `course_rag_search`

In `tools.py`:

- Loads the Chroma vector store.
- Creates a retriever with `k=2` similar chunks.
- Uses `.invoke(query)` (LangChain v0.2+).
- Concatenates the chunks and **truncates** the context (e.g. 800–1000 chars) to stay within the LLM’s context window and respond faster.

This tool is used inside the LangGraph nodes (concept and practice) to inject **course-specific context** into the prompts.

---

## 💬 Prompt Engineering

All prompts are defined in `prompts.py` using `PromptTemplate`.

### 1. 🧭 Router Prompt

Decides if a question is:

- `"faq"` – schedule, grading, logistics  
- `"concept"` – theory, explanations  
- `"practice"` – exercises, quiz-style questions

The router returns **exactly** one of: `faq`, `concept`, `practice`.

### 2. 📅 FAQ Prompt (Logistics & Evaluation)

- Contains **hard-coded** course logistics:
  - Schedule: e.g. *Monday 17h–19h, Wednesday 16h–19h*
  - Classroom: *PB-A02*
  - Full evaluation scheme:
    - First term: quizzes, project advances, assignments, midterms.
    - Second term: quizzes, advances, assignments, **final project 25%**.
- Explicitly instructs:
  - “Use ONLY this information for schedule and grading.”
  - “If something is not specified, say so.”

This stops the LLM from hallucinating fake schedules or percentages.

### 3. 🧠 Concept Prompt (RAG + Explanation)

- Used for conceptual questions (SQL, ER modeling, normalization, APIs, etc.).
- Receives:
  - `user_input`
  - `history` (conversation)
  - `retrieved_context` (RAG output)
- Instructions:
  - Use **only** retrieved course documents (syllabus, UC contents, quizzes).
  - Explain step by step in simple English.
  - Include small SQL examples when relevant.
  - If the documents don’t contain the answer, explicitly say so.

### 4. 🧾 JSON Prompt (Structured Output)

- Used to convert a draft answer into a structured JSON object:

```json
{
  "answer": "Full explanation in natural language",
  "key_points": ["Point 1", "Point 2"],
  "references": ["UC1 - Introduction to databases", "Evaluation - final project 25%"]
}
```

This is useful for:

- Post-processing answers.
- Showing structured information.
- Potential future UI features (e.g., bullet points, references section).

### 5. 🧪 Practice Prompt

- Reuses the JSON prompt but oriented to **practice questions**.
- Takes `user_input` and `retrieved_context` (which can be quiz files).
- Asks the LLM to generate quiz-style questions, answers and short explanations.

---

## 🔗 LangChain Design

All chains live in `chains.py` and use **`langchain-classic`**:

- `build_memory()` → `ConversationBufferMemory` to keep dialogue context.
- `build_router_chain(llm)` → `LLMChain` with `router_prompt`.
- `build_faq_chain(llm, memory)` → `LLMChain` with FAQ prompt + memory.
- `build_concept_sequential_chain(llm, memory)`:
  - **Version 1 (full):**  
    - First `LLMChain` generates a **draft explanation** from RAG + history.  
    - Second `LLMChain` converts it into structured JSON.
  - **Version 2 (fast):**  
    - Single `LLMChain` that directly produces JSON from RAG + question.
- `build_practice_chain(llm)` → `LLMChain` that outputs JSON with practice questions.

All chains are **LLM-agnostic**: they just receive a `llm` object (which comes from `llm_factory.py`).

---

## 🕸️ LangGraph Workflow

Defined in `graph.py`.

### 🧩 State definition

```python
class EduChatState(TypedDict, total=False):
    user_input: str
    mode: Literal["faq", "concept", "practice"]
    history: str
    retrieved_context: Optional[str]
    draft_answer: Optional[str]
    json_answer: Optional[str]
    final_answer: str
```

### 🧱 Nodes

1. `input_node` – pass-through, just sets the initial state.
2. `router_node` – calls `router_chain.invoke(...)` and sets `mode` in the state.
3. `faq_node` – calls `faq_chain` and sets `final_answer`.
4. `concept_node`:
   - Calls `course_rag_search(user_input)`.
   - Calls the concept chain (draft + JSON, or direct JSON).
   - Stores `retrieved_context`, `draft_answer`, `json_answer`, `final_answer`.
5. `practice_node`:
   - Calls `course_rag_search(user_input)`.
   - Calls the practice chain with `user_input`, `retrieved_context`, `draft_answer=""`.
   - Stores `json_answer`, `final_answer`.
6. `memory_node`:
   - Appends the latest turn to `history`:
     - `User: ...`
     - `Agent: ...`
7. `final_node` – no-op, just returns state.

### 🔀 Edges

- `START → input → router`
- `router` has **conditional edges**:
  - `faq` → `faq_node`
  - `concept` → `concept_node`
  - `practice` → `practice_node`
- Each of these flows into:
  - `faq_node → memory_node`
  - `concept_node → memory_node`
  - `practice_node → memory_node`
- Then:
  - `memory_node → final_node → END`

A global `InMemorySaver` checkpoint is used so that **`thread_id` = session id** keeps separate conversations.

---

## 🤖 LLM: Ollama `gemma3:4b`

The project uses an **open-source LLM running locally**, via **Ollama**:

- Model: `gemma3:4b` (or similar small-to-medium model).
- Integration: `langchain-ollama` (`ChatOllama`).
- Configured in:
  - `config.py` → `DEFAULT_CONFIG_LOW_TEMP`
  - `llm_factory.py` → `make_hf_llm` (which actually returns a `ChatOllama` instance).

Example config:

```python
DEFAULT_CONFIG_LOW_TEMP = LLMConfig(
    model_id="gemma3:4b",
    temperature=0.2,
    top_p=0.9,
    top_k=40,
    max_new_tokens=64,  # short answers for speed
)
```

This satisfies the project requirement of using **open LLMs (Ollama)** and allows fully local inference.

---

## 🧪 Parameters & Behavior

Some parameters you can easily tune:

- **Temperature** (`0.2` vs `0.7`):
  - 0.2 → more focused, less creative → ideal for syllabus/evaluation questions.
  - 0.7 → more creative → can be used for brainstorming practice questions.
- **Max new tokens**:
  - Lower (e.g. `64`) → faster, concise.
  - Higher (e.g. `256`) → more detailed but slower.
- **RAG context length**:
  - `MAX_CONTEXT_CHARS` in `tools.py`.
  - Fewer characters = faster, but less context.

You can run small experiments by changing these values and asking a set of 8–10 fixed questions, then comparing correctness, verbosity and hallucinations.

---

## 🖥️ Running the Project

> **Prerequisites:**
> - Python 3.10+  
> - Ollama installed and model `gemma3:4b` pulled  
> - (Optional) Git, VSCode  

### 1️⃣ Create virtual environment and install dependencies

From the project root:

```bash
python -m venv .venv

# PowerShell (Windows)
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2️⃣ Prepare RAG (index course documents)

```bash
cd src
python -m educhat.build_rag
```

You should see:

```text
Loaded N documents from data/raw
✅ Vector store built in data/processed/chroma
```

### 3️⃣ Run the chatbot in the console (CLI)

```bash
cd src
python -m educhat.cli
```

Example interaction:

```text
🔄 Compiling EduChatAgent graph, please wait...
✅ EduChatAgent ready.

Session id (e.g. andres): Andres

Type your questions about the DATABASES course.
Type 'exit' to quit.

You: What topics are covered in UC1?

[EduChatAgent] Generating answer, please wait...

EduChatAgent:
{
  "answer": "... detailed explanation ...",
  "key_points": [...],
  "references": ["UC1 - Fundamentals and Database Design"]
}
```

### 4️⃣ Start the API (FastAPI)

From `src`:

```bash
uvicorn educhat.api:app --reload
```

- Interactive docs: `http://127.0.0.1:8000/docs`
- Main endpoint: `POST http://127.0.0.1:8000/chat`

Example JSON body:

```json
{
  "session_id": "andres",
  "message": "How much does the final project cost?"
}
```

### 5️⃣ Open the Web UI

- File: `web/index.html`
- Just open it in your browser (double click).
- Make sure the API is running at `http://localhost:8000`.

The UI:

- Shows chat bubbles (user + bot).
- Sends messages via `fetch()` to `POST /chat`.
- Indicates status (“Ready”, “Thinking…”, network errors).

---

## ✅ How This Meets the Project Rubric

**LangChain Competencies**

- Prompt templates: `prompts.py` (router, FAQ, concept, JSON).
- Chains:
  - `SequentialChain` (concept explanation → JSON) or simplified single-chain mode.
  - Router chain, FAQ chain, practice chain.
- Memory:
  - `ConversationBufferMemory` in `chains.py`.
- Few-shot / structure:
  - Prompt patterns with examples and JSON output format.

**LangGraph Competencies**

- Graph with:
  - Router node (decides faq/concept/practice).
  - At least 4 nodes: input, router, faq_node, concept_node, practice_node, memory_node, final_node.
- State definition: `EduChatState` (TypedDict).
- Tool integration:
  - `course_rag_search()` as a retriever-based tool.
- Memory:
  - Graph-level state (`history`) and checkpointing via `InMemorySaver` + `thread_id`.

**LLM Competency**

- Open-source LLM via **Ollama (gemma3:4b)**.
- Parameter tuning (temperature, top_p, max_new_tokens) easily configurable in `config.py`.
- Observed behavior differences across configurations (for the written report).

**Agent Competency**

- Full pipeline:
  - **Environment**: syllabus + quizzes + evaluation.
  - **LLM Reasoning**: LangChain chains with prompts & memory.
  - **Tools/Actions**: RAG retrieval tool.
  - **Output**: structured JSON + natural language response.
- Evaluation:
  - Test questions from syllabus and old quizzes.
  - Manual error analysis of correct vs. incorrect answers.
- Comparison of strategies:
  - Without RAG vs. with RAG.
  - Single-step vs. two-step (draft + JSON) answers.

---

## 🔎 Possible Future Work

- Add **authentication** or per-student sessions.
- Store conversation logs in a real database instead of JSONL.
- Add **teacher mode** to generate new quiz questions from UC contents.
- Export answers / explanations as PDF cheat sheets.
- Integrate voice input/output for accessibility.
- Add more advanced evaluation (automatic grading of answers).

---

## 🙌 Credits

- **Author:** Andrés Basantes  
- **Advisor / Course:** Intelligent Agents – Databases, Yachay Tech  
- **Technologies:** Python, LangChain, LangGraph, Ollama, FastAPI, HTML/CSS/JS

> _“EduChatAgent is not just a chatbot; it is a course-specific assistant that understands the structure, content and evaluation of the Databases course, and helps students practice and clarify concepts in a grounded way.”_
