# 📚 AI Document Q&A Chatbot

An intelligent chatbot that can answer questions from uploaded PDF documents using Retrieval-Augmented Generation (RAG) technology.

## 🔗 Live Demo

**Try it now:** [AI Document Assistant](https://chatbot-project-prajwal.streamlit.app/)

## ✨ Features

- 💬 **Natural Conversations** - Chat with AI powered by Google Gemini Pro
- 📄 **PDF Document Upload** - Upload and analyze PDF documents
- 🔍 **Smart Search** - Uses vector embeddings for semantic search
- 🎯 **Context-Aware Answers** - Answers based on your specific documents
- 📊 **Document Chunking** - Intelligently splits documents for better analysis
- 💾 **Export Chat History** - Download conversation history as JSON
- ⚡ **Fast Processing** - Efficient text extraction and embedding generation

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
| **AI Model** | Google Gemini Pro |
| **RAG Framework** | LangChain |
| **Vector Database** | ChromaDB |
| **Embeddings** | HuggingFace (sentence-transformers) |
| **PDF Processing** | PyPDF2 |
| **Language** | Python 3.11 |

## 🏗️ Architecture
```
User uploads PDF → Extract text (PyPDF2)
                ↓
          Split into chunks (LangChain)
                ↓
     Generate embeddings (HuggingFace)
                ↓
      Store in vector DB (ChromaDB)
                ↓
   User asks question → Semantic search
                ↓
    Retrieve relevant chunks + Send to Gemini
                ↓
           Display answer with source
```

## 🚀 How It Works (RAG Explained)

**RAG (Retrieval-Augmented Generation)** combines retrieval and generation:

1. **Document Processing:**
   - PDF text is extracted and split into chunks (~1000 chars each)
   - Each chunk is converted to vector embeddings (numerical representations)
   - Embeddings are stored in ChromaDB for fast similarity search

2. **Question Answering:**
   - User's question is converted to an embedding
   - Vector database finds most similar chunks (semantic search)
   - Relevant chunks + question are sent to Gemini Pro
   - AI generates answer based on retrieved context

3. **Why RAG?**
   - Gemini doesn't inherently know your documents
   - RAG provides relevant context from your PDFs
   - Enables accurate, source-based answers

## 📦 Installation

### Prerequisites
- Python 3.8+
- Google Gemini API key ([Get it here](https://aistudio.google.com/app/apikey))

### Steps

1. **Clone the repository**
```bash
   git clone https://github.com/prajwal-gadkar/chatbot-project.git
   cd chatbot-project
```

2. **Create virtual environment**
```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Mac/Linux
   source venv/bin/activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Set up API key**
   
   Create a `.env` file in the root directory:
```env
   GOOGLE_API_KEY=your_gemini_api_key_here
```

5. **Run the app**
```bash
   streamlit run app.py
```

6. **Open in browser**
   
   The app will automatically open at `http://localhost:8501`

## 📖 Usage

### General Chat Mode
- Simply type your question and press "Send 🚀"
- Ask about any topic - the AI will respond using its general knowledge

### Document Q&A Mode
1. Click "Browse files" in the sidebar
2. Upload a PDF document
3. Wait for processing (first time may take 2-3 minutes to download embedding model)
4. Ask questions about your document!
5. The AI will answer based on the PDF content

### Export Chat History
- Click "💾 Export" in sidebar
- Download your conversation as JSON

## 🎯 Use Cases

- 📚 **Students:** Analyze research papers, textbooks, course materials
- 📊 **Professionals:** Query reports, documentation, manuals
- 📝 **Researchers:** Extract information from academic papers
- 💼 **Business:** Analyze contracts, policies, business documents

## 📁 Project Structure
```
chatbot-project/
├── app.py                 # Main application
├── requirements.txt       # Python dependencies
├── .env                   # API keys (not in git)
├── .gitignore            # Git ignore rules
├── README.md             # This file
├── .streamlit/
│   └── secrets.toml      # Deployment secrets (not in git)
└── screenshots/          # Project screenshots (optional)
```

## 🔑 Key Components

### PDF Processing
```python
def extract_text_from_pdf(pdf_file):
    # Extracts text from all pages
    # Handles multi-page documents
```

### Document Chunking
```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=200     # Overlap to maintain context
)
```

### Vector Search
```python
vector_store.similarity_search(question, k=3)
# Finds 3 most relevant chunks using cosine similarity
```

### RAG Prompt
```python
prompt = f"""Based on the following context, answer the question.
Context: {retrieved_chunks}
Question: {user_question}
"""
```

## 🚀 Deployment

### Deploy to Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add `GOOGLE_API_KEY` in secrets
5. Deploy!

Your app will be live at: `https://chatbot-project-prajwal.streamlit.app/`

## 🔮 Future Improvements

- [ ] Support multiple PDF uploads simultaneously
- [ ] Add DOCX and TXT file support
- [ ] Implement user authentication
- [ ] Add conversation memory across sessions
- [ ] Enable source highlighting in PDF
- [ ] Add summarization feature
- [ ] Support for multiple languages
- [ ] Implement feedback mechanism

## 🐛 Known Issues

- Scanned PDFs (images) won't work - needs OCR
- Very large PDFs (>100 pages) may take longer to process
- First-time embedding model download takes 2-3 minutes

## 👨‍💻 Author

**Prajwal**  
B.Tech CSE Student  

## 🙏 Acknowledgments

- Google Gemini for the AI model
- Streamlit for the amazing framework
- LangChain for RAG utilities
- HuggingFace for embeddings

---

