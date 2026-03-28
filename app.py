import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from datetime import datetime
import json

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv('GOOGLE_API_KEY')
if api_key:
    genai.configure(api_key=api_key)

# Page configuration
st.set_page_config(
    page_title="AI Document Q&A Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 5px;
        font-size: 13px;
        color: #555;
    }
    
    .chunk-preview {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        font-size: 12px;
        max-height: 100px;
        overflow-y: auto;
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
    
    .success-badge {
        background: #d4edda;
        color: #155724;
        padding: 8px 15px;
        border-radius: 20px;
        display: inline-block;
        margin: 5px 0;
        font-weight: bold;
    }
    
    .info-badge {
        background: #d1ecf1;
        color: #0c5460;
        padding: 8px 15px;
        border-radius: 20px;
        display: inline-block;
        margin: 5px 0;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* header {visibility: hidden;} */
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'model' not in st.session_state:
    if api_key:
        st.session_state.model = genai.GenerativeModel('gemini-2.5-flash')
    else:
        st.session_state.model = None

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

if 'current_pdf' not in st.session_state:
    st.session_state.current_pdf = None

if 'chunks_preview' not in st.session_state:
    st.session_state.chunks_preview = []

if 'processing_time' not in st.session_state:
    st.session_state.processing_time = 0

# Functions
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

def process_document(text, pdf_name):
    """Split text into chunks and create vector store"""
    start_time = datetime.now()
    
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
        metadatas=[{"source": pdf_name, "chunk_id": i} for i in range(len(chunks))]
    )
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    return vector_store, chunks, processing_time

def get_relevant_context(question, vector_store, k=3):
    """Retrieve relevant chunks for the question"""
    if vector_store is None:
        return None, []
    
    docs = vector_store.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context, docs

def export_chat_history():
    """Export chat history as JSON"""
    chat_data = {
        "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "document": st.session_state.current_pdf if st.session_state.pdf_processed else "No document",
        "total_messages": len(st.session_state.messages),
        "messages": st.session_state.messages
    }
    return json.dumps(chat_data, indent=2)

# Header
st.markdown("""
    <div class="main-header">
        <h1>📚 AI Document Q&A Assistant</h1>
        <p style="font-size: 20px; opacity: 0.95;">Upload PDFs & Get Intelligent Answers</p>
    </div>
""", unsafe_allow_html=True)

# Check API key
if not api_key:
    st.error("⚠️ **GOOGLE_API_KEY not found!** Please add it to your .env file.")
    st.stop()

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
            with st.spinner("📖 Processing document... This may take a moment."):
                try:
                    # Extract text
                    text = extract_text_from_pdf(uploaded_file)
                    
                    if not text.strip():
                        st.error("❌ No text found in PDF. It might be a scanned image.")
                    else:
                        # Process document
                        vector_store, chunks, proc_time = process_document(text, uploaded_file.name)
                        
                        # Update session state
                        st.session_state.vector_store = vector_store
                        st.session_state.pdf_processed = True
                        st.session_state.current_pdf = uploaded_file.name
                        st.session_state.chunks_preview = chunks[:5]  # Store first 5 chunks
                        st.session_state.processing_time = proc_time
                        
                        st.success(f"✅ **Processed:** {uploaded_file.name}")
                        st.info(f"📊 **Chunks created:** {len(chunks)}")
                        st.info(f"⏱️ **Processing time:** {proc_time:.2f}s")
                    
                except Exception as e:
                    st.error(f"❌ **Error:** {str(e)}")
    
    if st.session_state.pdf_processed:
        st.markdown("---")
        st.markdown("### 📑 Current Document")
        st.markdown(f'<div class="success-badge">✓ {st.session_state.current_pdf}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-badge">⏱️ Processed in {st.session_state.processing_time:.2f}s</div>', unsafe_allow_html=True)
        
        with st.expander("🔍 View Document Chunks"):
            st.markdown("**First 5 chunks:**")
            for i, chunk in enumerate(st.session_state.chunks_preview, 1):
                st.markdown(f"**Chunk {i}:**")
                st.markdown(f'<div class="chunk-preview">{chunk[:200]}...</div>', unsafe_allow_html=True)
        
        if st.button("🗑️ Remove Document", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.pdf_processed = False
            st.session_state.current_pdf = None
            st.session_state.chunks_preview = []
            st.rerun()
    
    st.markdown("---")
    
    st.markdown("## 💬 Chat Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("Export", use_container_width=True):
            if st.session_state.messages:
                chat_json = export_chat_history()
                st.download_button(
                    label="📥 Download",
                    data=chat_json,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.warning("No chat history to export")
    
    st.markdown("---")
    
    st.markdown("## 📊 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("💬 Messages", len(st.session_state.messages))
    with col2:
        user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.metric("❓ Questions", user_msgs)
    
    st.markdown("---")
    
    st.markdown("## 🎯 How to Use")
    st.info("""
        **Mode 1: General Chat**
        Ask any question without uploading a PDF
        
        **Mode 2: Document Q&A**
        1. Upload a PDF above
        2. Wait for processing
        3. Ask questions about it
        
        The AI retrieves relevant sections and answers based on your document!
    """)
    
    st.markdown("---")
    
    st.markdown("## 🛠️ Tech Stack")
    st.code("""
Frontend: Streamlit
AI Model: Google Gemini Pro
RAG: LangChain + ChromaDB
Embeddings: HuggingFace
PDF Parser: PyPDF2
Language: Python
    """, language="yaml")
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Prajwal**  \nB.Tech CSE Student  \n[GitHub](https://github.com/prajwal-gadkar)")

# Main chat area
if len(st.session_state.messages) == 0:
    # Welcome message
    if st.session_state.pdf_processed:
        st.markdown(f"""
            <div style="background: white; border-radius: 20px; padding: 40px; box-shadow: 0 10px 40px rgba(0,0,0,0.1); margin: 20px auto; max-width: 800px;">
                <h2 style="text-align: center; color: #667eea; margin-bottom: 20px;">
                    📄 Document Loaded & Ready!
                </h2>
                <p style="text-align: center; color: #666; font-size: 18px; margin-bottom: 30px;">
                    I've analyzed <strong>{st.session_state.current_pdf}</strong> and created a searchable knowledge base. Ask me anything about it!
                </p>
                <div style="text-align: center; background: #f8f9fa; padding: 20px; border-radius: 15px; margin: 20px 0;">
                    <p style="color: #888; margin-bottom: 10px;">💡 <strong>How it works:</strong></p>
                    <p style="color: #666; font-size: 14px; line-height: 1.8;">
                        Your document was split into chunks → Converted to embeddings → Stored in vector database<br>
                        When you ask a question, I search for relevant chunks and use them to generate accurate answers!
                    </p>
                </div>
                <div style="text-align: center; margin-top: 30px;">
                    <p style="color: #888; margin-bottom: 15px;">Example questions:</p>
                    <div style="background: #667eea; color: white; padding: 12px 20px; border-radius: 20px; display: inline-block; margin: 5px;">
                        "Summarize this document"
                    </div>
                    <div style="background: #667eea; color: white; padding: 12px 20px; border-radius: 20px; display: inline-block; margin: 5px;">
                        "What are the key points?"
                    </div>
                    <div style="background: #667eea; color: white; padding: 12px 20px; border-radius: 20px; display: inline-block; margin: 5px;">
                        "Find information about [topic]"
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background: white; border-radius: 20px; padding: 40px; box-shadow: 0 10px 40px rgba(0,0,0,0.1); margin: 20px auto; max-width: 800px;">
                <h2 style="text-align: center; color: #667eea; margin-bottom: 20px;">
                    👋 Welcome to AI Document Q&A!
                </h2>
                <p style="text-align: center; color: #666; font-size: 18px; margin-bottom: 30px;">
                    I can help you in two ways:
                </p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 30px 0;">
                    <div style="background: #f8f9fa; padding: 25px; border-radius: 15px; text-align: center;">
                        <div style="font-size: 40px; margin-bottom: 15px;">💬</div>
                        <h3 style="color: #667eea; margin-bottom: 10px;">General Chat</h3>
                        <p style="color: #666; font-size: 14px;">Ask me anything! I can explain concepts, help with coding, solve problems, and more.</p>
                    </div>
                    <div style="background: #f8f9fa; padding: 25px; border-radius: 15px; text-align: center;">
                        <div style="font-size: 40px; margin-bottom: 15px;">📄</div>
                        <h3 style="color: #667eea; margin-bottom: 10px;">Document Analysis</h3>
                        <p style="color: #666; font-size: 14px;">Upload a PDF and I'll answer questions based on its content using advanced RAG technology.</p>
                    </div>
                </div>
                <div style="text-align: center; margin-top: 40px;">
                    <p style="color: #888; margin-bottom: 15px;">Try asking:</p>
                    <div style="background: #667eea; color: white; padding: 12px 20px; border-radius: 20px; display: inline-block; margin: 5px;">
                        "What is machine learning?"
                    </div>
                    <div style="background: #667eea; color: white; padding: 12px 20px; border-radius: 20px; display: inline-block; margin: 5px;">
                        "Explain Python classes"
                    </div>
                    <div style="background: #667eea; color: white; padding: 12px 20px; border-radius: 20px; display: inline-block; margin: 5px;">
                        "Help me debug code"
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
    with st.spinner("🤔 Analyzing and generating response..."):
        try:
            # Check if PDF is uploaded
            if st.session_state.pdf_processed and st.session_state.vector_store:
                # RAG mode - get relevant context
                context, docs = get_relevant_context(user_input, st.session_state.vector_store)
                
                if context:
                    # Create prompt with context
                    prompt = f"""Based on the following context from the document, answer the question accurately and comprehensively.

Context:
{context}

Question: {user_input}

Instructions:
- Answer based primarily on the context provided
- Be specific and cite relevant information
- If the answer is not fully covered in the context, mention what information is available and what might be missing
- Keep the response clear and well-structured"""
                    
                    response = st.session_state.model.generate_content(prompt)
                    bot_response = response.text
                    source_info = f"Retrieved from {len(docs)} relevant sections of {st.session_state.current_pdf}"
                else:
                    bot_response = "I couldn't find relevant information in the document to answer that question."
                    source_info = None
                
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
            error_msg = f"⚠️ Error generating response: {str(e)}\n\nPlease try again or rephrase your question."
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
        Built with ❤️ using Streamlit, Google Gemini Pro & RAG Technology | 
        <a href="https://github.com/prajwal-gadkar/chatbot-project" style="color: white; text-decoration: none;">View on GitHub</a>
    </div>
""", unsafe_allow_html=True)