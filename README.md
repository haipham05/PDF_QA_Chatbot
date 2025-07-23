# Document and Video QA Chatbot: Deep, Accessible Technical Documentation

A comprehensive, modular Retrieval-Augmented Generation (RAG) chatbot for question answering over PDF documents and YouTube video transcripts. This document explains every component in detail, using analogies and plain language, so anyone can understand how and why it works—even without prior AI or NLP knowledge.

---

## How the Full RAG Workflow Operates (Step-by-Step)

This system is built as a **graph workflow** (using LangGraph), where each step is a node, and the flow can branch based on decisions. Here’s how a user question is processed from start to finish:

### 1. **User Input**
- The user uploads a PDF or enters a YouTube URL, and then asks a question.

### 2. **Content Loading & Chunking**
- The document or transcript is loaded and split into overlapping chunks for better retrieval.

### 3. **Embedding & Vector Store**
- Each chunk is embedded (turned into a vector) and stored in FAISS for fast similarity search.

### 4. **Graph Workflow Begins**

#### **A. Retrieve Node**
- The system embeds the user’s question and retrieves the top-k most similar chunks from FAISS.
- These chunks are then reranked by Flashrank for better relevance.

#### **B. Grade Documents Node**
- The system checks: "Do the retrieved and reranked chunks actually contain enough information to answer the question?"
- This is done by prompting the LLM as a **relevance grader**. The LLM sees the question and the context, and must answer "yes" (sufficient) or "no" (insufficient).

#### **C. Route Decision Node**
- If the LLM grader says "yes":
    - The workflow routes to the **Generate Node**.
- If the LLM grader says "no":
    - The workflow ends, and the user is told: "Sorry, I couldn't find enough relevant information to answer your question."

#### **D. Generate Node**
- The LLM is prompted with the question and the reranked, relevant context.
- It generates a concise answer, using only the provided context. If the answer is not in the context, it is instructed to say so.

#### **E. End Node**
- The answer (or a message about insufficient information) is returned to the user.

### **Why This Matters**
- **Grading and Routing** ensure the system only answers when it has enough evidence, reducing hallucination.
- **Graph-based design** makes the workflow modular, explainable, and easy to extend (e.g., you could add more nodes for summarization, citation, etc.).

---

## Table of Contents
- [Project Overview](#project-overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Component-by-Component Deep Dive (with Analogies)](#component-by-component-deep-dive-with-analogies)
  - [1. Content Ingestion](#1-content-ingestion)
  - [2. Text Chunking](#2-text-chunking)
  - [3. Embedding Generation](#3-embedding-generation)
  - [4. Vector Store: FAISS](#4-vector-store-faiss)
  - [5. Initial Retrieval](#5-initial-retrieval)
  - [6. Neural Reranking: Flashrank](#6-neural-reranking-flashrank)
  - [7. Relevance Grading](#7-relevance-grading)
  - [8. Answer Generation (LLM)](#8-answer-generation-llm)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Pros and Cons of Each Technique](#pros-and-cons-of-each-technique)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Project Overview
This project lets you ask questions about the content of a PDF or a YouTube video, and get answers grounded in the actual content. It uses a modern AI pipeline called Retrieval-Augmented Generation (RAG), which means it first finds the most relevant parts of the content, then asks an AI language model to answer your question using only those parts.

---

## Pipeline Architecture

```mermaid
graph TD
    A[User Input: PDF or YouTube URL] --> B[Content Loading]
    B --> C[Text Chunking]
    C --> D[Embedding Generation]
    D --> E[FAISS Vector Store]
    E --> F[Initial Retrieval (kNN)]
    F --> G[Neural Reranking (Flashrank)]
    G --> H[Relevance Grading (LLM)]
    H --> I{Relevant?}
    I -- Yes --> J[Answer Generation (LLM)]
    I -- No --> K[Notify User: Not enough info]
```

---

## Component-by-Component Deep Dive (with Analogies)

### 1. Content Ingestion
- **PDF**: The system reads your PDF, page by page, and extracts the text. Think of it like a librarian scanning every page of a book and making a digital copy.
- **YouTube**: The system fetches the transcript (subtitles) of a YouTube video using `youtube_transcript_api`. This is like getting the script of a movie, so you can read what was said without watching.
- **Why**: This step turns your content into plain text, which is the raw material for all later steps.

### 2. Text Chunking
- **What**: The text is split into overlapping "chunks" (e.g., 1000 characters each, with 200 characters of overlap).
- **Why**: Imagine you want to find a quote in a book. If you only look at whole chapters, you might miss it. If you look at every sentence, you might get lost. Chunks are like flipping through a book in manageable, overlapping sections, so you don't miss anything important that crosses a page break.
- **How**: The `RecursiveCharacterTextSplitter` tries to split at natural boundaries (paragraphs, sentences), but will break up long sections if needed. Overlap ensures that information at the edge of one chunk is also in the next.

### 3. Embedding Generation
- **What**: Each chunk is turned into a list of numbers (an "embedding") that captures its meaning.
- **Analogy**: Imagine you have a magical machine that can turn any sentence into a unique color in a huge rainbow. Sentences with similar meanings get similar colors. In reality, the "color" is a point in a high-dimensional space (e.g., 384 numbers), and similar meanings are close together.
- **How**: The embedding model (e.g., Llama3 via Ollama) is a neural network trained to map similar texts to nearby points. This is called "semantic embedding".
- **Why**: Computers can't compare meanings directly, but they can compare numbers. Embeddings let us use math to find similar content.

### 4. Vector Store: FAISS
- **What**: FAISS is a library that stores all the embeddings and lets us quickly find which ones are closest to a new query.
- **Analogy**: Imagine a giant map where every chunk is a city, and the embedding is its coordinates. If you want to find the city most like your query, you look for the nearest city on the map.
- **How**: FAISS supports different ways to search:
  - **IndexFlatL2**: Looks at every city and finds the closest (exact, but slower for huge maps).
  - **IndexIVFFlat**: Divides the map into regions, only searches the most likely regions (faster, a bit less accurate).
  - **IndexHNSWFlat**: Uses a "shortcut network" to jump quickly between cities (very fast for huge maps).
- **Distance**: Usually, "closeness" is measured by Euclidean distance (straight-line) or cosine similarity (angle between vectors).
- **Why**: This lets us do "semantic search"—find the most relevant chunks for any question, even if the words don't match exactly.

### 5. Initial Retrieval
- **What**: When you ask a question, it is also embedded (turned into a point on the map). FAISS finds the k (e.g., 5) closest chunks.
- **Analogy**: It's like asking, "Which cities are nearest to this new city?"
- **Why**: This step narrows down the content to just the most relevant parts, making the next steps faster and more accurate.

### 6. Neural Reranking: Flashrank
- **What**: Flashrank is a "cross-encoder" reranker. It takes the top-k chunks and scores them again, but this time it looks at the question and each chunk together, not separately.
- **Analogy**: Imagine you have a panel of expert judges. The first judge (FAISS) quickly picks the top 5 candidates. The second judge (Flashrank) interviews each candidate with the actual question and gives a more thoughtful score.
- **How**: Flashrank uses a neural network (e.g., MiniLM, BGE) that takes both the question and chunk as input, and outputs a relevance score. It is ONNX-optimized, meaning it's very fast even though it's more accurate.
- **Why**: The initial retrieval might miss subtle connections. Reranking lets the system "think harder" about which chunks really answer the question.

### 7. Relevance Grading
- **What**: The system checks if the reranked chunks actually contain enough information to answer the question.
- **How**: It asks the LLM: "Given this question and these chunks, can you answer? Respond 'yes' or 'no'."
- **Analogy**: It's like a teacher checking if the textbook pages you found really have the answer, before letting you write your essay.
- **Why**: This prevents the system from making up answers (hallucinating) when the content isn't actually there.

### 8. Answer Generation (LLM)
- **What**: The LLM (e.g., Llama3) is given the question and the reranked, relevant chunks. It is told to answer only using the provided text.
- **How**: The prompt is carefully designed to instruct the LLM to be concise, accurate, and to admit when it doesn't know.
- **Analogy**: It's like a student writing an essay, but only allowed to use the textbook pages provided.
- **Why**: This keeps answers grounded in the source material, reducing the risk of hallucination.

---

## Pipeline Walkthrough
1. **User uploads PDF or enters YouTube URL**
2. **Text is loaded and split into overlapping chunks**
3. **Chunks are embedded using Ollama's embedding model**
4. **Embeddings are stored in a FAISS index (IndexFlatL2 by default)**
5. **On question, query is embedded and top-k chunks are retrieved via kNN**
6. **Flashrank reranks the retrieved chunks using a cross-encoder**
7. **LLM grades if the context is sufficient**
8. **If relevant, LLM generates answer; otherwise, user is notified**

---

## Pros and Cons of Each Technique

### **RecursiveCharacterTextSplitter**
- **Pros**: Preserves context, avoids breaking sentences, supports overlap
- **Cons**: Chunk size/overlap must be tuned for best results

### **Ollama Embeddings**
- **Pros**: Local, privacy-preserving, supports multiple models
- **Cons**: Requires local resources, model quality may vary

### **FAISS (IndexFlatL2, IVF, HNSW)**
- **Pros**: Fast, scalable, supports both exact and approximate search
- **Cons**: RAM usage for large datasets, approximate search may miss some neighbors

### **Flashrank (Cross-Encoder Reranking)**
- **Pros**: State-of-the-art accuracy, ONNX-optimized for speed, corrects retrieval errors
- **Cons**: Adds compute cost, limited by initial retrieval recall

### **LLM-based Grading and Answering**
- **Pros**: Reduces hallucination, ensures answers are grounded
- **Cons**: LLM may still err if context is ambiguous or insufficient

### **youtube_transcript_api**
- **Pros**: No API key needed, works for most public videos
- **Cons**: Fails for private/region-locked/no-caption videos

---

## Project Structure
- `app.py` : Streamlit web app for PDF/YouTube QA
- `PDF.py` : Terminal-based PDF QA chatbot
- `video.py` : Terminal-based YouTube QA chatbot
- `requirements.txt` : Python dependencies

---

## Installation & Usage
See earlier sections for detailed instructions.

---

## Troubleshooting
- **Ollama not running**: Ensure `ollama serve` is running and the model is pulled
- **No transcript found**: Some YouTube videos lack captions or are restricted
- **Large PDFs/videos**: May require significant RAM and time
- **Streamlit errors**: Restart the app if you encounter port/session issues

---

## References
- [LangChain Documentation](https://python.langchain.com/)
- [Ollama](https://ollama.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Flashrank](https://python.langchain.com/docs/integrations/retrievers/flashrank-reranker/)
- [youtube_transcript_api](https://github.com/jdepoix/youtube-transcript-api) 