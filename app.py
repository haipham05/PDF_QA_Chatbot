import streamlit as st
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
import tempfile
import os
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document

st.set_page_config(page_title="Document/Video QABot", layout="wide")
st.title("📄📹 Document/Video QABot")

st.markdown("""
A simple RAG chatbot for answering questions based on your PDF document or YouTube video.\
Powered by LangChain, Ollama, FAISS, and Flashrank reranker.
""")

# Sidebar for model selection and settings
with st.sidebar:
    st.header("Settings")
    ollama_model = st.text_input("Ollama Model", value="llama3")
    chunk_size = st.number_input("Chunk Size", min_value=256, max_value=4096, value=1000, step=128)
    chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=1024, value=200, step=50)

# Input type selector
input_type = st.radio("Select input type", ["PDF", "YouTube Video"])

uploaded_file = None
video_url = None
docs = None
source_type = None

if input_type == "PDF":
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            st.success(f"Loaded {len(docs)} pages from your PDF.")
            source_type = "pdf"
        except Exception as e:
            st.error(f"Error loading document: {e}")
            st.stop()
elif input_type == "YouTube Video":
    video_url = st.text_input("Enter YouTube video URL")
    if video_url:
        try:
            # Validate URL format
            if not ("youtube.com/watch?v=" in video_url or "youtu.be/" in video_url):
                st.error("Invalid YouTube URL format. Please provide a valid YouTube URL.")
                st.stop()
            # Extract video ID
            if "youtube.com/watch?v=" in video_url:
                video_id = video_url.split("v=")[-1].split("&")[0]
            else:
                video_id = video_url.split("youtu.be/")[-1].split("?")[0]
            # Fetch transcript using youtube_transcript_api
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                # Prefer English transcript
                transcript = None
                for t in transcript_list:
                    if t.language_code.startswith('en'):
                        transcript = t.fetch()
                        break
                if transcript is None:
                    # Fallback to first available
                    transcript = transcript_list.find_transcript([t.language_code for t in transcript_list]).fetch()
                def get_text(item):
                    if isinstance(item, dict):
                        return item.get('text', '')
                    return getattr(item, 'text', '')
                text = " ".join([get_text(item) for item in transcript])
                docs = [Document(page_content=text, metadata={"source": video_url, "title": f"YouTube Video {video_id}", "language": transcript_list._manually_created_transcripts[0].language_code if transcript_list._manually_created_transcripts else 'unknown'})]
            except Exception as e:
                st.error(f"No transcript found for this video. It may be private, region-locked, or have no captions. Error: {e}")
                st.stop()
            if not docs or not docs[0].page_content.strip():
                st.error("No transcript found for this video. It may be private, region-locked, or have no captions.")
                st.stop()
            st.success(f"Loaded transcript for: {docs[0].metadata.get('title', 'Unknown title')}")
            source_type = "youtube"
        except Exception as e:
            st.error(f"Error loading video transcript: {e}")
            st.stop()

if docs:
    # Split document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    st.info(f"Content split into {len(chunks)} chunks.")

    # Initialize models
    try:
        llm = ChatOllama(model=ollama_model, temperature=0)
        embeddings = OllamaEmbeddings(model=ollama_model)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        compressor = FlashrankRerank()
        rerank_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
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
        documents = rerank_retriever.invoke(question)
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
            if source_type == "pdf":
                return {"generation": "I'm sorry, I don't have enough information from the documents to answer your question.", "question": question, "documents": documents}
            else:
                return {"generation": "I'm sorry, I don't have enough information from the video transcript to answer your question.", "question": question, "documents": documents}
        if source_type == "pdf":
            sys_prompt = """You are a helpful assistant.\nUse the provided 'Text' to answer the 'Question'.\nIf you cannot answer based on the provided text, explicitly state that you do not have enough information from the document.\nKeep your answer concise and directly address the question.\n----------------\nText: {context}"""
        else:
            sys_prompt = """You are a helpful assistant.\nUse the provided 'Text' from a video transcript to answer the 'Question'.\nIf you cannot answer based on the provided transcript, explicitly state that you do not have enough information from the video.\nKeep your answer concise and directly address the question.\n----------------\nText: {context}"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
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

    # For YouTube, show a summary after loading
    if source_type == "youtube":
        st.subheader("Video Summary")
        full_transcript = "\n".join([doc.page_content for doc in docs])
        max_length = 8000
        if len(full_transcript) > max_length:
            full_transcript = full_transcript[:max_length] + "..."
            st.info("Video is long, summarizing first part only.")
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at creating comprehensive video summaries.\nYour task is to analyze the provided video transcript and create a well-structured summary.\n\nPlease provide:\n1. **Main Topic**: What is this video about?\n2. **Key Points**: List the 3-5 most important points discussed\n3. **Summary**: A concise 2-3 paragraph overview\n4. **Duration Estimate**: Approximate video length based on transcript\n\nKeep the summary informative but concise. Focus on the main ideas and valuable insights."""),
            ("user", "Please summarize this video transcript:\n\n{transcript}")
        ])
        summary_chain = summary_prompt | llm | StrOutputParser()
        try:
            summary = summary_chain.invoke({"transcript": full_transcript})
            st.markdown(summary)
        except Exception as e:
            st.warning(f"Error generating summary: {e}")

    # Chat interface
    st.subheader("Ask a question about your content:")
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
                answer = "Sorry, I couldn't find enough relevant information in your content to answer that question."
            else:
                answer = "Sorry, an unexpected issue occurred. Please try again."
            st.session_state.chat_history.append((user_question, answer))

    # Display chat history
    for q, a in st.session_state.get("chat_history", [])[::-1]:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Chatbot:** {a}")

    # Clean up temp file
    if input_type == "PDF":
        os.remove(tmp_path)
else:
    if input_type == "PDF":
        st.info("Please upload a PDF document to get started.")
    else:
        st.info("Please enter a YouTube video URL to get started.")
