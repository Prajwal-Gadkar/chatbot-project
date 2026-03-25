# AI Chatbot with Document Q&A

An intelligent chatbot built with Google Gemini AI that can answer questions and analyze uploaded documents.

## Features
- Natural language conversations
- PDF document upload and analysis
- Beautiful, responsive UI
- Chat history
- Real-time responses

## Tech Stack
- **Frontend:** Streamlit
- **AI Model:** Google Gemini Pro
- **Backend:** Python
- **Vector Database:** ChromaDB (for document search)

## Installation

1. Clone the repository
```bash
git clone https://github.com/prajwal-gadkar/chatbot-project.git
cd chatbot-project
```

2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Create `.env` file and add your Gemini API key
```
GOOGLE_API_KEY=your_api_key_here
```

5. Run the app
```bash
streamlit run app.py
```

## Usage
1. Open the app in your browser (opens automatically)
2. Type your question in the input box
3. Upload PDF documents to ask questions about them
4. View chat history in the sidebar

## Project Structure
```
chatbot-project/
├── app.py              # Main application
├── .env                # API keys (not in git)
├── .gitignore          # Git ignore rules
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## Future Improvements
- [ ] Multiple document support
- [ ] Chat export functionality
- [ ] User authentication
- [ ] Conversation memory across sessions
- [ ] Support for more file types (DOCX, TXT)

## Author
Prajwal - B.Tech CSE Student

