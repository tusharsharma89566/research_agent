import streamlit as st
import os
import tempfile
from typing import List, Dict, Any
from dotenv import load_dotenv

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

# Hugging Face imports
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# Load environment variables
load_dotenv()

class AIResearchAssistant:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self.memory = None
        
    def initialize_components(self):
        """Initialize all AI components"""
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize LLM
        self.llm = self._setup_llm()
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def _setup_llm(self):
        """Setup Hugging Face LLM"""
        model_name = "google/flan-t5-base"
        
        try:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Create text generation pipeline
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.7,
                do_sample=True,
            )
            
            # Wrap in LangChain pipeline
            llm = HuggingFacePipeline(pipeline=pipe)
            return llm
            
        except Exception as e:
            st.error(f"Error loading LLM: {str(e)}")
            # Fallback to a simpler model
            return self._setup_fallback_llm()
    
    def _setup_fallback_llm(self):
        """Setup fallback LLM if main model fails"""
        try:
            pipe = pipeline(
                "text-generation",
                model="gpt2",
                max_length=256,
                temperature=0.7,
                pad_token_id=50256
            )
            return HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            st.error(f"Error loading fallback LLM: {str(e)}")
            return None
    
    def load_documents(self, uploaded_files) -> List[Document]:
        """Load and process uploaded PDF documents"""
        documents = []
        
        for uploaded_file in uploaded_files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            try:
                # Load PDF
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
                documents.extend(docs)
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {str(e)}")
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
        
        return documents
    
    def process_documents(self, documents: List[Document]):
        """Process documents and create vector store"""
        if not documents:
            st.warning("No documents to process.")
            return
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        
        # Create QA chain
        self._create_qa_chain()
        
        st.success(f"Processed {len(documents)} documents into {len(splits)} chunks.")
    
    def _create_qa_chain(self):
        """Create the conversational QA chain"""
        if not self.vectorstore or not self.llm:
            return
        
        # Create custom prompt template
        prompt_template = """You are a helpful research assistant. Use the following context to answer the user's question. 
        If you don't know the answer based on the context, just say "I don't have enough information to answer that question."
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Create retrieval chain
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
    
    def ask_question(self, question: str) -> str:
        """Ask a question and get an answer"""
        if not self.qa_chain:
            return "Please upload and process documents first."
        
        try:
            response = self.qa_chain({"question": question})
            return response["answer"]
        except Exception as e:
            return f"Error processing question: {str(e)}"

def main():
    st.set_page_config(
        page_title="AI Research Assistant",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 AI-Powered Research Assistant")
    st.markdown("Upload your documents and ask questions using RAG with Hugging Face and LangChain!")
    
    # Initialize session state
    if "assistant" not in st.session_state:
        st.session_state.assistant = AIResearchAssistant()
        st.session_state.assistant.initialize_components()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload PDF documents to create your knowledge base"
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    documents = st.session_state.assistant.load_documents(uploaded_files)
                    st.session_state.assistant.process_documents(documents)
        
        st.markdown("---")
        st.header("📊 System Info")
        
        if st.session_state.assistant.vectorstore:
            st.success("✅ Vector store ready")
        else:
            st.info("📋 Upload documents to get started")
        
        if st.session_state.assistant.llm:
            st.success("✅ LLM loaded")
        else:
            st.error("❌ LLM not loaded")
    
    # Main chat interface
    st.header("💬 Chat with your Documents")
    
    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**You:** {question}")
            st.markdown(f"**Assistant:** {answer}")
            st.markdown("---")
    
    # Question input
    question = st.text_input(
        "Ask a question about your documents:",
        placeholder="What is the main topic discussed in the documents?",
        key="question_input"
    )
    
    col1, col2 = st.columns([1, 5])
    
    with col1:
        ask_button = st.button("Ask", type="primary")
    
    with col2:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.assistant.memory.clear()
    
    # Process question
    if ask_button and question:
        if not st.session_state.assistant.vectorstore:
            st.error("Please upload and process documents first!")
        else:
            with st.spinner("Thinking..."):
                answer = st.session_state.assistant.ask_question(question)
                st.session_state.chat_history.append((question, answer))
    
    # Instructions
    with st.expander("📖 How to use this app"):
        st.markdown("""
        1. **Upload Documents**: Use the sidebar to upload PDF files
        2. **Process Documents**: Click "Process Documents" to create the knowledge base
        3. **Ask Questions**: Type your questions in the chat interface
        4. **Get Answers**: The AI will provide answers based on your documents
        
        **Features:**
        - RAG (Retrieval-Augmented Generation) for accurate answers
        - Conversational memory to maintain context
        - Multiple PDF support
        - Hugging Face open-source models
        """)

if __name__ == "__main__":
    main()
