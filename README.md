
# research_agent

## Overview

**research_agent** is an AI-powered research assistant built with Python, Streamlit, LangChain, and Hugging Face. It enables users to upload PDF documents and ask questions about their content using Retrieval-Augmented Generation (RAG). The assistant leverages advanced language models and embeddings to provide insightful answers with conversational memory and multi-document support.

---

## Features

- **AI-Powered Q&A**: Ask questions about your uploaded documents and receive concise, context-aware answers.
- **Retrieval-Augmented Generation (RAG)**: Combines your documents with large language models for accurate, up-to-date responses.
- **Conversational Memory**: Maintains chat history for context-aware conversations.
- **Multiple PDF Support**: Upload and query multiple documents simultaneously.
- **Open-Source Models**: Uses Hugging Face models for both embeddings and language understanding.
- **Streamlit UI**: Intuitive web interface for document upload and chat.

---

## How It Works

1. **Upload Documents**: Use the sidebar to upload one or more PDF files.
2. **Process Documents**: The assistant splits the documents into manageable chunks and creates a searchable vector store using embeddings.
3. **Ask Questions**: Enter your research questions in the chat interface. The assistant retrieves relevant information and generates answers using a conversational RAG pipeline.
4. **Chat History**: Review previous questions and answers in the main interface.

---

## Tech Stack

- **Python**
- **Streamlit** for the web interface
- **LangChain** for chaining LLMs and document retrieval
- **Hugging Face Transformers** for language models and embeddings
- **FAISS** for vector search and similarity retrieval

---

## Getting Started

### Prerequisites

- Python 3.8+
- `pip` for installing dependencies

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tusharsharma89566/research_agent.git
   cd research_agent
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Create a `.env` file if needed (for API keys or configuration).

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser:**  
   Go to `http://localhost:8501` to use the assistant.

---

## Usage

1. **Upload PDFs** using the sidebar.
2. Click **Process Documents** to build the knowledge base.
3. **Ask questions** in the chat box.  
   Example: `What is the main topic discussed in these documents?`
4. Review the answers and chat history.

---

## Example

![Screenshot](screenshot.png)

---

## Customization

- **Model Selection:** The app uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings and `google/flan-t5-base` for LLM. You can modify `app.py` to use different Hugging Face models.
- **Chunk Size:** Adjust document chunking in the `process_documents` method for larger/smaller splits.

---

## License

This project is open source. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Streamlit](https://github.com/streamlit/streamlit)
