import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Page configuration
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Chat message styling */
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
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Header styling */
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
    
    /* Input styling */
    .stTextInput input {
        border-radius: 25px;
        border: 2px solid #667eea;
        padding: 15px 20px;
        font-size: 16px;
        background: white;
    }
    
    .stTextInput input:focus {
        border-color: #764ba2;
        box-shadow: 0 0 10px rgba(118, 75, 162, 0.3);
    }
    
    /* Button styling */
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
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.95);
    }
    
    /* Welcome card */
    .welcome-card {
        background: white;
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin: 20px auto;
        max-width: 800px;
        animation: scaleIn 0.5s ease-out;
    }
    
    @keyframes scaleIn {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* Hide Streamlit branding */
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

# Header
st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Chat Assistant</h1>
        <p style="font-size: 20px; opacity: 0.95;">Powered by Google Gemini Pro</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 💬 Chat Controls")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
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
    
    st.markdown("## ℹ️ About")
    st.info("""
        **AI Chat Assistant** uses Google's Gemini Pro model to provide intelligent, context-aware responses.
        
        **Features:**
        ✨ Natural conversations  
        🚀 Fast responses  
        💡 Context awareness  
        🎨 Beautiful interface
    """)
    
    st.markdown("---")
    
    st.markdown("## 🛠️ Tech Stack")
    st.code("""
    Frontend: Streamlit
    AI Model: Gemini Pro
    Language: Python
    """, language="yaml")
    
    st.markdown("---")
    
    st.markdown("## 👨‍💻 Developer")
    st.markdown("**Prajwal**  \nB.Tech CSE Student")

# Main chat area
if len(st.session_state.messages) == 0:
    # Welcome message when chat is empty
    st.markdown("""
        <div class="welcome-card">
            <h2 style="text-align: center; color: #667eea; margin-bottom: 20px;">
                👋 Welcome! How can I help you today?
            </h2>
            <p style="text-align: center; color: #666; font-size: 18px; margin-bottom: 30px;">
                I'm an AI assistant powered by Google's Gemini Pro. Ask me anything!
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 30px;">
                <div style="background: #f8f9fa; padding: 20px; border-radius: 15px; text-align: center;">
                    <div style="font-size: 30px; margin-bottom: 10px;">💡</div>
                    <div style="color: #667eea; font-weight: bold;">Explain Concepts</div>
                    <div style="color: #999; font-size: 14px; margin-top: 5px;">Ask me to explain any topic</div>
                </div>
                <div style="background: #f8f9fa; padding: 20px; border-radius: 15px; text-align: center;">
                    <div style="font-size: 30px; margin-bottom: 10px;">💻</div>
                    <div style="color: #667eea; font-weight: bold;">Code Help</div>
                    <div style="color: #999; font-size: 14px; margin-top: 5px;">Get coding assistance</div>
                </div>
                <div style="background: #f8f9fa; padding: 20px; border-radius: 15px; text-align: center;">
                    <div style="font-size: 30px; margin-bottom: 10px;">🎯</div>
                    <div style="color: #667eea; font-weight: bold;">Problem Solving</div>
                    <div style="color: #999; font-size: 14px; margin-top: 5px;">Work through challenges</div>
                </div>
            </div>
            <div style="margin-top: 40px; text-align: center;">
                <p style="color: #888; margin-bottom: 15px;">Try asking:</p>
                <div style="background: #667eea; color: white; padding: 12px 20px; border-radius: 20px; display: inline-block; margin: 5px;">
                    "What is artificial intelligence?"
                </div>
                <div style="background: #667eea; color: white; padding: 12px 20px; border-radius: 20px; display: inline-block; margin: 5px;">
                    "Explain Python decorators"
                </div>
                <div style="background: #667eea; color: white; padding: 12px 20px; border-radius: 20px; display: inline-block; margin: 5px;">
                    "Tell me a joke"
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

# Input area
st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns([6, 1])

with col1:
    user_input = st.text_input(
        "Message",
        placeholder="Type your message here...",
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
            response = st.session_state.model.generate_content(user_input)
            bot_response = response.text
            
            # Add bot response
            st.session_state.messages.append({
                "role": "assistant",
                "content": bot_response
            })
            
        except Exception as e:
            error_msg = f"⚠️ Error: {str(e)}\n\nPlease check your API key and try again."
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })
    
    # Refresh to show new messages
    st.rerun()

# Footer info
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: white; opacity: 0.7; font-size: 14px;">
        Built with ❤️ using Streamlit & Google Gemini
    </div>
""", unsafe_allow_html=True)