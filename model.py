from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="llama3", temperature=0)
embeddings = OllamaEmbeddings(model="llama3")

file_path = input("Input your document path here: ")
try:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"Successfully loaded {len(docs)} pages from {file_path}")
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
print(f"Document split into {len(chunks)} chunks.")

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
print("Vector store created and retriever initialized.")

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
    print("\n---NODE: RETRIEVE DOCUMENTS---")
    question = state["question"]
    documents = retriever.invoke(question)
    print(f"Retrieved {len(documents)} documents.")
    return {"documents": documents, "question": question}

def grade_documents(state: GraphState):
    """
    Grades the relevance of the retrieved documents to the question.
    """
    print("\n---NODE: GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]

    if not documents:
        print("No documents retrieved for grading. Marking as 'not_relevant'.")
        return {"document_grade": "not_relevant", "question": question, "documents": documents}

    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert in document relevance grading.
         Your mission is to assess if the provided 'Context' contains enough relevant information to answer the 'Question'.
         Respond 'yes' if the Context is relevant or contains key information, and 'no' if it is not relevant or insufficient.
         Only respond 'yes' or 'no'. Your response should be a single word."""),
        ("user", "Question: {question}\nContext: {context}")
    ])

    grading_chain = grade_prompt | llm | StrOutputParser()

    context_for_grading = "\n---\n".join([doc.page_content for doc in documents])

    try:
        grade = grading_chain.invoke({"question": question, "context": context_for_grading}).strip().lower()
    except Exception as e:
        print(f"Error during document grading: {e}. Defaulting to 'not_relevant'.")
        grade = "no"

    document_grade = "relevant" if grade == "yes" else "not_relevant"
    print(f"Documents graded as: {document_grade}")
    return {"document_grade": document_grade, "question": question, "documents": documents}


def generate(state: GraphState):
    """
    Generate answer using LLM and retrieved documents
    """
    print("\n---NODE: GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]

    if not documents:
        print("No documents available for generation.")
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
    print("Answer generated.")
    return {"documents": documents, "question": question, "generation": generation}


def route_decision(state: GraphState):
    """
    This function acts as the conditional router.
    It takes the current state and decides the next step based on 'document_grade'.
    """
    print("\n---NODE: ROUTE DECISION---")
    document_grade = state["document_grade"]

    if document_grade == "relevant":
        print("Routing to 'generate' (documents are relevant).")
        return "generate"
    else:
        print("Routing to 'end' (documents not relevant or insufficient).")
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
        "generate": "generate",          # If route_decision returns "generate"
        "end_with_no_answer": END        # If route_decision returns "end_with_no_answer"
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

    inputs = {"question": user_question}
    try:
        final_state = None
        for s in app.stream(inputs):
            print(f"Current state: {s}")
            print("---")
            final_state = s

        if final_state:
            if "generate" in final_state and "generation" in final_state["generate"]:
                print(f"Chatbot: {final_state['generate']['generation']}")
            elif "retrieve" in final_state:
                print("Chatbot: Sorry, I couldn't find enough relevant information in your document to answer that question.")
            else:
                print("Chatbot: Sorry, an unexpected issue occurred. Please try again.")

    except Exception as e:
        print(f"Error during chatbot interaction: {e}")
        print("Chatbot: Sorry, I can't respond to your query now.")