# src/educhat/chains.py

from langchain_classic.chains import LLMChain, SequentialChain
from langchain_classic.memory import ConversationBufferMemory

from .prompts import (
    router_prompt,
    faq_prompt,
    concept_prompt,
    structured_json_prompt,
)


def build_memory() -> ConversationBufferMemory:
    """
    Conversation memory for the chatbot.
    """
    return ConversationBufferMemory(
        memory_key="history",
        input_key="user_input",
        return_messages=False,
    )


def build_router_chain(llm) -> LLMChain:
    """
    Chain that decides which mode to use: faq / concept / practice.
    """
    return LLMChain(
        llm=llm,
        prompt=router_prompt,
        output_key="mode",
    )


def build_faq_chain(llm, memory: ConversationBufferMemory) -> LLMChain:
    """
    Chain used for FAQ-style questions (logistics, grading, schedule).
    """
    return LLMChain(
        llm=llm,
        prompt=faq_prompt,
        memory=memory,
        output_key="answer",
    )


def build_concept_sequential_chain(llm, memory: ConversationBufferMemory) -> SequentialChain:
    """
    SequentialChain for CONCEPT questions.
    Step 1: generate a pedagogical draft answer using RAG context.
    Step 2: convert it into a structured JSON object (answer + key_points + references).
    """
    draft_chain = LLMChain(
        llm=llm,
        prompt=concept_prompt,
        memory=memory,
        output_key="draft_answer",
    )

    json_chain = LLMChain(
        llm=llm,
        prompt=structured_json_prompt,
        output_key="json_answer",
    )

    chain = SequentialChain(
        chains=[draft_chain, json_chain],
        input_variables=["user_input", "history", "retrieved_context"],
        output_variables=["draft_answer", "json_answer"],
        verbose=False,
    )
    return chain


def build_practice_chain(llm) -> LLMChain:
    """
    Chain used for PRACTICE questions.

    It expects:
      - "user_input": what the student wants to practice
      - "retrieved_context": course documents with examples / quizzes
    """
    practice_prompt = structured_json_prompt

    return LLMChain(
        llm=llm,
        prompt=practice_prompt,
        output_key="json_answer",
    )
