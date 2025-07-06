from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import YoutubeLoader
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

def get_youtube_url():
    """Get YouTube URL from user input dialog"""
    root = tk.Tk()
    root.withdraw()
    
    # Create a simple input dialog
    from tkinter import simpledialog
    
    youtube_url = simpledialog.askstring(
        "YouTube Video",
        "Enter YouTube URL:",
        parent=root
    )
    root.destroy()
    return youtube_url

def load_youtube_video(youtube_url):
    """Load YouTube video transcript using YoutubeLoader with fallback options"""
    if not youtube_url:
        print("No URL provided. Exiting.")
        return None
    
    # Clean and validate URL
    if "youtube.com/watch?v=" not in youtube_url and "youtu.be/" not in youtube_url:
        print("Invalid YouTube URL format. Please provide a valid YouTube URL.")
        return None
    
    print("Attempting to load YouTube video...")
    
    # Method 1: Try with language preferences
    for language in ['en', 'en-US', None]:
        try:
            print(f"Trying with language: {language if language else 'auto-detect'}")
            loader = YoutubeLoader.from_youtube_url(
                youtube_url,
                add_video_info=True,
                language=language if language else ['en'],
                continue_on_failure=True
            )
            docs = loader.load()
            if docs and docs[0].page_content.strip():
                print(f"✓ Successfully loaded video: {docs[0].metadata.get('title', 'Unknown title')}")
                # Split documents after successful loading
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(docs)
                return chunks
        except Exception as e:
            print(f"  Failed with language {language}: {str(e)[:100]}...")
            continue
    
    # Method 2: Try with transcript list approach
    try:
        print("Trying alternative transcript method...")
        from youtube_transcript_api._api import YouTubeTranscriptApi
        import re
        
        # Extract video ID from URL
        video_id = None
        if "youtube.com/watch?v=" in youtube_url:
            video_id = youtube_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in youtube_url:
            video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
        
        if video_id:
            # Get available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to get English transcript first
            for transcript in transcript_list:
                if transcript.language_code in ['en', 'en-US', 'en-GB']:
                    transcript_data = transcript.fetch()
                    text = " ".join([item.text for item in transcript_data])
                    
                    from langchain_core.documents import Document
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": youtube_url,
                            "title": f"YouTube Video {video_id}",
                            "language": transcript.language_code
                        }
                    )
                    print(f"✓ Successfully loaded transcript in {transcript.language_code}")
                    
                    # Split documents
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_documents([doc])
                    return chunks
            
            # If no English, try the first available
            available_transcripts = list(transcript_list)
            if available_transcripts:
                transcript = available_transcripts[0]
                transcript_data = transcript.fetch()
                text = " ".join([item.text for item in transcript_data])
                
                from langchain_core.documents import Document
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": youtube_url,
                        "title": f"YouTube Video {video_id}",
                        "language": transcript.language_code
                    }
                )
                print(f"✓ Successfully loaded transcript in {transcript.language_code}")
                
                # Split documents
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents([doc])
                return chunks
                
    except Exception as e:
        print(f"Alternative method failed: {str(e)[:100]}...")
    
    print("❌ Failed to load video transcript. Possible reasons:")
    print("  - Video has no captions/transcript available")
    print("  - Video is private or restricted")
    print("  - Regional restrictions")
    print("  - Invalid URL")
    print("\nTry with a different YouTube video that has captions enabled.")
    return None

# Get YouTube URL and load video
youtube_url = get_youtube_url()
docs = load_youtube_video(youtube_url)

def generate_video_summary(docs):
    """
    Automatically generate a summary of the video content
    """
    print("\n🔄 Generating video summary...")
    
    # Combine all document content
    full_transcript = "\n".join([doc.page_content for doc in docs])
    
    # Limit transcript length for processing (if too long, take first part)
    max_length = 8000  # Adjust based on your model's context window
    if len(full_transcript) > max_length:
        full_transcript = full_transcript[:max_length] + "..."
        print("⚠️  Video is long, summarizing first part only.")
    
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at creating comprehensive video summaries.
         Your task is to analyze the provided video transcript and create a well-structured summary.
         
         Please provide:
         1. **Main Topic**: What is this video about?
         2. **Key Points**: List the 3-5 most important points discussed
         3. **Summary**: A concise 2-3 paragraph overview
         4. **Duration Estimate**: Approximate video length based on transcript
         
         Keep the summary informative but concise. Focus on the main ideas and valuable insights."""),
        ("user", "Please summarize this video transcript:\n\n{transcript}")
    ])
    
    summary_chain = summary_prompt | llm | StrOutputParser()
    
    try:
        summary = summary_chain.invoke({"transcript": full_transcript})
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"

if not docs:
    exit()

# Generate and display automatic summary
print("=" * 60)
summary = generate_video_summary(docs)
print(f"\n📝 VIDEO SUMMARY")
print("=" * 60)
print(summary)
print("=" * 60)

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's question
        generation: LLM generation
        documents: List of documents (chunks)
        document_grade: str
    """
    question: str
    generation: str
    documents: List[str]
    document_grade: str

def retrieve(state: GraphState):
    """
    Retrieve documents from the vectorstore
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
        return {"generation": "I'm sorry, I don't have enough information from the video transcript to answer your question.", "question": question, "documents": documents}

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant.
         Use the provided 'Text' from a video transcript to answer the 'Question'.
         If you cannot answer based on the provided transcript, explicitly state that you do not have enough information from the video.
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

# Build the workflow
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

print("\n🤖 YOUTUBE VIDEO CHATBOT is ready!")
print("💡 Ask questions about the video content above, or type 'exit' to quit.")

while True:
    user_question = input("\n❓ User: ")
    if user_question.lower() == 'exit':
        break

    inputs: GraphState = {"question": user_question, "generation": "", "documents": [], "document_grade": ""}
    try:
        final_state = None
        for s in app.stream(inputs):
            final_state = s

        if final_state:
            if "generate" in final_state and "generation" in final_state["generate"]:
                print(f"🤖 Chatbot: {final_state['generate']['generation']}")
            elif "retrieve" in final_state:
                print("🤖 Chatbot: Sorry, I couldn't find enough relevant information in the video transcript to answer the question.")
            else:
                print("🤖 Chatbot: Sorry, an unexpected issue occurred. Please try again.")

    except Exception as e:
        print("🤖 Chatbot: Sorry, I can't respond to your query now.")