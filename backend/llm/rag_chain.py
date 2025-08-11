# backend/llm/rag_chain.py

from __future__ import annotations
import os
from typing import Dict, Optional

import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.schema import BaseRetriever

# Preferred Together path via Langfuse OpenAI drop-in
try:
    from backend.graphs.monitor import get_traced_openai_together_client
except Exception:
    get_traced_openai_together_client = None  # type: ignore

# Fallback Together path via LangChain
try:
    from langchain_together import Together  # pip install langchain-together
except Exception:
    Together = None  # type: ignore

# Enhanced RAG prompt based on 2025 best practices
_RAG_PROMPT = PromptTemplate.from_template(
    """You are an expert academic assistant specializing in educational document analysis. Your task is to answer questions about academic modules, courses, and educational content using ONLY the provided context documents.

INSTRUCTIONS:
1. **Strict Context Adherence**: Base your answer EXCLUSIVELY on the information provided in the context below
2. **No External Knowledge**: Do not use any information from your training data or general knowledge
3. **Explicit Source References**: When possible, reference specific parts of the context in your answer
4. **Structured Responses**: Organize your answer clearly with appropriate formatting
5. **Uncertainty Handling**: If the context doesn't contain enough information to fully answer the question, state what you can answer and explicitly mention what information is missing

RESPONSE GUIDELINES:
- For factual questions: Provide direct, concise answers with specific details from the context
- For complex questions: Break down your response into logical components
- For incomplete information: State "Based on the provided context, I can only tell you..." and list what is available
- For completely unanswerable questions: Respond with "I don't have enough information in the provided context to answer this question."

CONTEXT:
{context}

QUESTION: {question}

ANSWER (following the above guidelines):"""
)

# Alternative prompt for academic metadata extraction specifically
_ACADEMIC_RAG_PROMPT = PromptTemplate.from_template(
    """You are a specialist in academic program documentation. Answer the user's question using ONLY the information from the provided academic documents.

ANALYSIS APPROACH:
1. **Scan Context**: Carefully review all provided text for relevant information
2. **Extract Details**: Identify specific facts, requirements, procedures, or policies mentioned
3. **Synthesize Response**: Organize the information into a clear, comprehensive answer
4. **Acknowledge Gaps**: If information is incomplete, clearly state what is missing

RESPONSE FORMAT:
- Lead with the most direct answer to the question
- Include relevant details and specifics from the context  
- Use bullet points or numbered lists for complex information
- Quote directly when exact wording is important
- End with any limitations or missing information

ACADEMIC CONTEXT:
{context}

USER QUESTION: {question}

DETAILED RESPONSE:"""
)

# Chain-of-thought prompt for complex reasoning
_COT_RAG_PROMPT = PromptTemplate.from_template(
    """You are an expert academic consultant. Answer the question using the provided context through step-by-step analysis.

REASONING PROCESS:
1. **Context Analysis**: What relevant information is available in the provided documents?
2. **Question Breakdown**: What specific aspects of the question can be addressed?
3. **Evidence Gathering**: What specific facts, dates, requirements, or details support the answer?
4. **Synthesis**: How do these pieces of information combine to answer the question?
5. **Gaps Identification**: What information would be needed for a complete answer that isn't available?

CONTEXT DOCUMENTS:
{context}

QUESTION: {question}

STEP-BY-STEP ANALYSIS:
1. Context Analysis: 
2. Question Breakdown:
3. Evidence Gathering:
4. Synthesis:
5. Final Answer:"""
)

def _invoke_llm(model_name: str, prompt: str) -> str:
    if not os.getenv("TOGETHER_API_KEY"):
        raise EnvironmentError("TOGETHER_API_KEY is missing – export it or put it in .env")

    # Preferred: Langfuse OpenAI drop-in to Together
    if get_traced_openai_together_client is not None:
        client = get_traced_openai_together_client()
        if client is not None:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert academic assistant that provides precise, context-based answers to educational queries. Always ground your responses in the provided context and acknowledge limitations clearly."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,  # Increased for more detailed responses
                temperature=0.1,  # Lower for more consistent, factual responses
                top_p=0.9,
            )
            return (resp.choices[0].message.content or "").strip()

    # Fallback: langchain_together
    if Together is None:
        raise RuntimeError("Neither Langfuse OpenAI drop-in nor langchain_together is available")
    llm = Together(model=model_name, max_tokens=1024, temperature=0.1, top_p=0.9)
    return llm.invoke(prompt).strip()

class _SimpleRAGChain:
    """Enhanced RAG wrapper with improved prompting strategies."""

    def __init__(
        self,
        *,
        llm_model: str,
        retriever: BaseRetriever,
        prompt: Optional[PromptTemplate] = None,
        prompt_type: str = "standard",  # "standard", "academic", or "cot"
    ) -> None:
        self._model = llm_model
        self._retriever = retriever
        
        # Select appropriate prompt based on type
        if prompt is not None:
            self._prompt = prompt
        elif prompt_type == "academic":
            self._prompt = _ACADEMIC_RAG_PROMPT
        elif prompt_type == "cot":
            self._prompt = _COT_RAG_PROMPT
        else:
            self._prompt = _RAG_PROMPT

    def run(self, inputs: Dict[str, str]) -> str:
        question: str = inputs["query"]
        docs = self._retriever.get_relevant_documents(question)
        
        if len(docs) == 0:
            st.warning("⚠️ No relevant documents found for your question")
            return "I don't have any relevant information in my knowledge base to answer your question."
        
        st.info(f"🔎 Retrieved {len(docs)} relevant chunks from vector store")
        
        # Enhanced context formatting with document boundaries
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[Document {i}]\n{doc.page_content}")
        
        context = "\n\n".join(context_parts)
        prompt_str = self._prompt.format(context=context, question=question)
        return _invoke_llm(self._model, prompt_str)

def build_rag_chain(
    *,
    llm_model: str,
    retriever: BaseRetriever,
    prompt: Optional[PromptTemplate] = None,
    prompt_type: str = "standard",  # New parameter for prompt selection
    return_sources: bool = False,  # kept for parity
) -> _SimpleRAGChain:
    """
    Build a RAG chain with enhanced prompting.
    
    Args:
        llm_model: The model to use
        retriever: Document retriever
        prompt: Custom prompt template (optional)
        prompt_type: Type of prompt to use ("standard", "academic", "cot")
        return_sources: Legacy parameter for compatibility
    """
    return _SimpleRAGChain(
        llm_model=llm_model, 
        retriever=retriever, 
        prompt=prompt,
        prompt_type=prompt_type
    )
