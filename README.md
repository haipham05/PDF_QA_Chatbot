# DocumentQABot

A simple RAG (Retrieval-Augmented Generation) chatbot for answering questions based on the content of a PDF document using LangChain, Ollama, and Streamlit.

## Features
- Upload a PDF document and ask questions about its content.
- Uses LangChain's document loaders, text splitters, and vector stores (FAISS).
- Embeddings and LLM powered by Ollama (e.g., llama3 model).
- Document relevance grading to ensure answers are based on relevant content.
- Modular graph workflow for retrieval, grading, and answer generation.

## Requirements
- Python 3.8+
- [Ollama](https://ollama.com/) (with the `llama3` model pulled)
- Streamlit
- langchain, langchain_community, langgraph, langchain_core
- PyPDFLoader, langchain_text_splitters

## Installation
1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd DocumentQABot
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install and run Ollama, then pull the required model:
   ```bash
   ollama pull llama3
   ollama serve
   ```

## Usage
### Command Line
Run the chatbot in the terminal:
```bash
python model.py
```
You will be prompted to enter the path to your PDF document, then you can ask questions interactively.

### Streamlit Web App
```bash
streamlit run app.py
```

## Project Structure
- `model.py`: Main chatbot logic and workflow.
- `1.ipynb`: Example notebook for experimentation.
- `Real_Time_Deepfake_Detection_DPL_Sample_Project.pdf`: Example PDF document.

## Notes
- Make sure the Ollama server is running and the required model is available.
- For large PDFs, initial processing may take some time.

