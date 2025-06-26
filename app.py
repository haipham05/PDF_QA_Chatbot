import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import tempfile
import os

st.set_page_config(page_title="DocumentQABot", layout="wide")
st.title("📄 DocumentQABot")

st.markdown("""
A simple RAG chatbot for answering questions based on your PDF document.\
Powered by LangChain, Ollama, and FAISS.
""")

# Sidebar for model selection and settings
with st.sidebar:
    st.header("Settings")
    ollama_model = st.text_input("Ollama Model", value="llama3")
    chunk_size = st.number_input("Chunk Size", min_value=256, max_value=4096, value=1000, step=128)
    chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=1024, value=200, step=50)

# File uploader
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        st.success(f"Loaded {len(docs)} pages from your PDF.")
    except Exception as e:
        st.error(f"Error loading document: {e}")
        st.stop()

    # Split document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    st.info(f"Document split into {len(chunks)} chunks.")

    # Initialize models
    try:
        llm = ChatOllama(model=ollama_model, temperature=0)
        embeddings = OllamaEmbeddings(model=ollama_model)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        st.error(f"Error initializing models or vectorstore: {e}")
        st.stop()

    class GraphState(TypedDict):
        question: str
        generation: str
        documents: List[str]
        document_grade: str

    def retrieve(state: GraphState):
        question = state["question"]
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}

    def grade_documents(state: GraphState):
        question = state["question"]
        documents = state["documents"]
        if not documents:
            return {"document_grade": "not_relevant", "question": question, "documents": documents}
        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in document relevance grading.\nYour mission is to assess if the provided 'Context' contains enough relevant information to answer the 'Question'.\nRespond 'yes' if the Context is relevant or contains key information, and 'no' if it is not relevant or insufficient.\nOnly respond 'yes' or 'no'. Your response should be a single word."""),
            ("user", "Question: {question}\nContext: {context}")
        ])
        grading_chain = grade_prompt | llm | StrOutputParser()
        context_for_grading = "\n---\n".join([doc.page_content for doc in documents])
        try:
            grade = grading_chain.invoke({"question": question, "context": context_for_grading}).strip().lower()
        except Exception:
            grade = "no"
        document_grade = "relevant" if grade == "yes" else "not_relevant"
        return {"document_grade": document_grade, "question": question, "documents": documents}

    def generate(state: GraphState):
        question = state["question"]
        documents = state["documents"]
        if not documents:
            return {"generation": "I'm sorry, I don't have enough information from the documents to answer your question.", "question": question, "documents": documents}
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant.\nUse the provided 'Text' to answer the 'Question'.\nIf you cannot answer based on the provided text, explicitly state that you do not have enough information from the document.\nKeep your answer concise and directly address the question.\n----------------\nText: {context}"""),
            ("user", "{question}")
        ])
        rag_chain = prompt | llm | StrOutputParser()
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def route_decision(state: GraphState):
        document_grade = state["document_grade"]
        if document_grade == "relevant":
            return "generate"
        else:
            return "end_with_no_answer"

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        route_decision,
        {
            "generate": "generate",
            "end_with_no_answer": END
        }
    )
    workflow.add_edge("generate", END)
    app = workflow.compile()

    # Chat interface
    st.subheader("Ask a question about your document:")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Your question", key="user_question")
    ask_button = st.button("Ask")

    if ask_button and user_question:
        with st.spinner("Generating answer..."):
            inputs = {"question": user_question}
            final_state = None
            for s in app.stream(inputs):
                final_state = s
            if final_state and "generate" in final_state and "generation" in final_state["generate"]:
                answer = final_state["generate"]["generation"]
            elif final_state and "retrieve" in final_state:
                answer = "Sorry, I couldn't find enough relevant information in your document to answer that question."
            else:
                answer = "Sorry, an unexpected issue occurred. Please try again."
            st.session_state.chat_history.append((user_question, answer))

    # Display chat history
    for q, a in st.session_state.get("chat_history", [])[::-1]:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Chatbot:** {a}")

    # Clean up temp file
    os.remove(tmp_path)
else:
    st.info("Please upload a PDF document to get started.")
