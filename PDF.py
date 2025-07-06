from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import tkinter as tk
from tkinter import filedialog

llm = ChatOllama(model="llama3", temperature=0)
embeddings = OllamaEmbeddings(model="llama3")

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select a PDF file",
    filetypes=[("PDF files", "*.pdf")]
)
if not file_path:
    print("No file selected. Exiting.")
    exit()

try:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
except Exception as e:
    print(f"Error loading document: {e}")
    print("Please ensure the file path is correct and the file is a valid PDF.")
    exit()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
chunks = text_splitter.split_documents(docs)

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's question
        generation: LLM generation
        documents: List of documents (chunks)
        # Add a field for document grade to pass between nodes
        document_grade: str
    """
    question: str
    generation: str
    documents: List[str]
    document_grade: str

def retrieve(state: GraphState):
    """
    Retrieve documents from vectorstore
    """
    question = state["question"]
    documents = retriever.invoke(question)
    # Extract page_content from Document objects to match GraphState type
    document_contents = [doc.page_content for doc in documents]
    return {"documents": document_contents, "question": question}

def grade_documents(state: GraphState):
    """
    Grades the relevance of the retrieved documents to the question.
    """
    question = state["question"]
    documents = state["documents"]

    if not documents:
        return {"document_grade": "not_relevant", "question": question, "documents": documents}

    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert in document relevance grading.
         Your mission is to assess if the provided 'Context' contains enough relevant information to answer the 'Question'.
         Respond 'yes' if the Context is relevant or contains key information, and 'no' if it is not relevant or insufficient.
         Only respond 'yes' or 'no'. Your response should be a single word."""),
        ("user", "Question: {question}\nContext: {context}")
    ])

    grading_chain = grade_prompt | llm | StrOutputParser()

    context_for_grading = "\n---\n".join(documents)

    try:
        grade = grading_chain.invoke({"question": question, "context": context_for_grading}).strip().lower()
    except Exception as e:
        grade = "no"

    document_grade = "relevant" if grade == "yes" else "not_relevant"
    return {"document_grade": document_grade, "question": question, "documents": documents}


def generate(state: GraphState):
    """
    Generate answer using LLM and retrieved documents
    """
    question = state["question"]
    documents = state["documents"]

    if not documents:
        return {"generation": "I'm sorry, I don't have enough information from the documents to answer your question.", "question": question, "documents": documents}

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant.
         Use the provided 'Text' to answer the 'Question'.
         If you cannot answer based on the provided text, explicitly state that you do not have enough information from the document.
         Keep your answer concise and directly address the question.
         ----------------
         Text: {context}"""),
        ("user", "{question}")
    ])

    rag_chain = prompt | llm | StrOutputParser()
    
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def route_decision(state: GraphState):
    """
    This function acts as the conditional router.
    It takes the current state and decides the next step based on 'document_grade'.
    """
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

print("\n--- CHATBOT is ready ---")
print("Press 'exit' to quit.")

while True:
    user_question = input("\nUser: ")
    if user_question.lower() == 'exit':
        break

    inputs: GraphState = {"question": user_question, "generation": "", "documents": [], "document_grade": ""}
    try:
        final_state = None
        for s in app.stream(inputs):
            final_state = s

        if final_state:
            if "generate" in final_state and "generation" in final_state["generate"]:
                print(f"Chatbot: {final_state['generate']['generation']}")
            elif "retrieve" in final_state:
                print("Chatbot: Sorry, I couldn't find enough relevant information in your document to answer the question.")
            else:
                print("Chatbot: Sorry, an unexpected issue occurred. Please try again.")

    except Exception as e:
        print("Chatbot: Sorry, I can't respond to your query now.")