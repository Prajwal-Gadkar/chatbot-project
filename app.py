import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import tempfile

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Page configuration
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (same beautiful styling)
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        margin-left: 20%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        animation: slideInRight 0.3s ease-out;
    }
    
    .bot-message {
        background: #f0f2f6;
        color: #1e1e1e;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        margin-right: 20%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        animation: slideInLeft 0.3s ease-out;
    }
    
    .source-box {
        background: #e8f4f8;
        border-left: 4px solid #667eea;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
        font-size: 14px;
        color: #555;
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .main-header {
        text-align: center;
        color: white;
        padding: 30px;
        margin-bottom: 20px;
        animation: fadeIn 0.8s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .main-header h1 {
        font-size: 3em;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .stTextInput input {
        border-radius: 25px;
        border: 2px solid #667eea;
        padding: 15px 20px;
        font-size: 16px;
        background: white;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        padding: 12px 30px;
        border: none;
        font-weight: bold;
        font-size: 16px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.95);
    }
    
    .upload-section {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'model' not in st.session_state:
    st.session_state.model = genai.GenerativeModel('gemini-2.5-flash')

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

if 'current_pdf' not in st.session_state:
    st.session_state.current_pdf = None

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to process document
def process_document(text, pdf_name):
    """Split text into chunks and create vector store"""
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create vector store
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=[{"source": pdf_name} for _ in chunks]
    )
    
    return vector_store, len(chunks)

# Function to get relevant context
def get_relevant_context(question, vector_store, k=3):
    """Retrieve relevant chunks for the question"""
    if vector_store is None:
        return None
    
    docs = vector_store.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context, docs

# Header
st.markdown("""
    <div class="main-header">
        <h1>📚 AI Document Assistant</h1>
        <p style="font-size: 20px; opacity: 0.95;">Upload PDFs & Ask Questions</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📄 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document to analyze"
    )
    
    if uploaded_file is not None:
        if st.session_state.current_pdf != uploaded_file.name:
            with st.spinner("📖 Processing document..."):
                try:
                    # Extract text
                    text = extract_text_from_pdf(uploaded_file)
                    
                    # Process document
                    vector_store, num_chunks = process_document(text, uploaded_file.name)
                    
                    # Update session state
                    st.session_state.vector_store = vector_store
                    st.session_state.pdf_processed = True
                    st.session_state.current_pdf = uploaded_file.name
                    
                    st.success(f"✅ Processed: {uploaded_file.name}")
                    st.info(f"📊 Created {num_chunks} text chunks")
                    
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
    
    if st.session_state.pdf_processed:
        st.markdown("### 📑 Current Document")
        st.success(f"**{st.session_state.current_pdf}**")
        
        if st.button("🗑️ Remove Document", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.pdf_processed = False
            st.session_state.current_pdf = None
            st.rerun()
    
    st.markdown("---")
    
    st.markdown("## 💬 Chat Controls")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("## 📊 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", len(st.session_state.messages))
    with col2:
        user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.metric("Questions", user_msgs)
    
    st.markdown("---")
    
    st.markdown("## 🎯 How to Use")
    st.info("""
        **Without PDF:**
        Ask general questions
        
        **With PDF:**
        1. Upload PDF above
        2. Wait for processing
        3. Ask questions about the document
        
        The AI will answer based on the PDF content!
    """)
    
    st.markdown("---")
    
    st.markdown("## 🛠️ Tech Stack")
    st.code("""
    Frontend: Streamlit
    AI: Gemini Pro
    RAG: LangChain + ChromaDB
    Embeddings: HuggingFace
    """, language="yaml")

# Main chat area
if len(st.session_state.messages) == 0:
    # Welcome message
    if st.session_state.pdf_processed:
        st.markdown(f"""
            <div style="background: white; border-radius: 20px; padding: 40px; box-shadow: 0 10px 40px rgba(0,0,0,0.1); margin: 20px auto; max-width: 800px;">
                <h2 style="text-align: center; color: #667eea; margin-bottom: 20px;">
                    📄 Document Ready!
                </h2>
                <p style="text-align: center; color: #666; font-size: 18px; margin-bottom: 30px;">
                    I've analyzed <strong>{st.session_state.current_pdf}</strong>. Ask me anything about it!
                </p>
                <div style="text-align: center;">
                    <p style="color: #888; margin-bottom: 15px;">Example questions:</p>
                    <div style="background: #667eea; color: white; padding: 12px 20px; border-radius: 20px; display: inline-block; margin: 5px;">
                        "Summarize this document"
                    </div>
                    <div style="background: #667eea; color: white; padding: 12px 20px; border-radius: 20px; display: inline-block; margin: 5px;">
                        "What are the main points?"
                    </div>
                    <div style="background: #667eea; color: white; padding: 12px 20px; border-radius: 20px; display: inline-block; margin: 5px;">
                        "Explain [specific topic]"
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background: white; border-radius: 20px; padding: 40px; box-shadow: 0 10px 40px rgba(0,0,0,0.1); margin: 20px auto; max-width: 800px;">
                <h2 style="text-align: center; color: #667eea; margin-bottom: 20px;">
                    👋 Welcome! How can I help?
                </h2>
                <p style="text-align: center; color: #666; font-size: 18px; margin-bottom: 30px;">
                    Upload a PDF to ask questions about it, or chat normally!
                </p>
                <div style="text-align: center;">
                    <p style="color: #888; margin-bottom: 15px;">Try asking:</p>
                    <div style="background: #667eea; color: white; padding: 12px 20px; border-radius: 20px; display: inline-block; margin: 5px;">
                        "What is machine learning?"
                    </div>
                    <div style="background: #667eea; color: white; padding: 12px 20px; border-radius: 20px; display: inline-block; margin: 5px;">
                        "Explain REST APIs"
                    </div>
                    <div style="background: #667eea; color: white; padding: 12px 20px; border-radius: 20px; display: inline-block; margin: 5px;">
                        "Tell me about Python"
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
else:
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f'<div class="user-message">👤 <strong>You:</strong><br>{message["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="bot-message">🤖 <strong>Assistant:</strong><br>{message["content"]}</div>',
                unsafe_allow_html=True
            )
            
            # Show source if available
            if "source" in message and message["source"]:
                st.markdown(
                    f'<div class="source-box">📎 <strong>Source:</strong> {message["source"]}</div>',
                    unsafe_allow_html=True
                )

# Input area
st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns([6, 1])

with col1:
    user_input = st.text_input(
        "Message",
        placeholder="Type your question here...",
        label_visibility="collapsed",
        key="user_input"
    )

with col2:
    send_button = st.button("Send 🚀", use_container_width=True)

# Handle message sending
if send_button and user_input:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Get AI response
    with st.spinner("🤔 Thinking..."):
        try:
            # Check if PDF is uploaded
            if st.session_state.pdf_processed and st.session_state.vector_store:
                # RAG mode - get relevant context
                context, docs = get_relevant_context(user_input, st.session_state.vector_store)
                
                # Create prompt with context
                prompt = f"""Based on the following context from the document, answer the question.
                
Context:
{context}

Question: {user_input}

Answer the question based on the context provided. If the answer is not in the context, say so."""
                
                response = st.session_state.model.generate_content(prompt)
                bot_response = response.text
                source_info = f"Based on {st.session_state.current_pdf}"
                
            else:
                # Normal chat mode
                response = st.session_state.model.generate_content(user_input)
                bot_response = response.text
                source_info = None
            
            # Add bot response
            st.session_state.messages.append({
                "role": "assistant",
                "content": bot_response,
                "source": source_info
            })
            
        except Exception as e:
            error_msg = f"⚠️ Error: {str(e)}"
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "source": None
            })
    
    st.rerun()

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: white; opacity: 0.7; font-size: 14px;">
        Built with ❤️ using Streamlit, Gemini Pro & RAG
    </div>
""", unsafe_allow_html=True)