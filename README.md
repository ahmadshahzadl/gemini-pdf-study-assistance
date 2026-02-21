# Gemini PDF Study Assistant 📚

A smart, AI-powered study companion that helps students and learners interact with their study materials using natural language questions and answers.

## 🎯 Project Overview

**Gemini PDF Study Assistant** is an intelligent document analysis tool built with Generative AI that allows users to upload PDF documents (textbooks, notes, resumes, etc.) and ask questions about the content. The application uses **Google's Gemini** (free tier) and **LangChain** to retrieve relevant context and provide accurate, helpful answers tailored to different document types.

### Key Features:

- 📄 **PDF Upload & Analysis** - Upload any PDF document for analysis
- 🤖 **AI-Powered Q&A** - Ask questions and get intelligent answers using Gemini (free)
- 📝 **Document Summarization** - Get a quick 5-point summary of your documents
- 🎓 **Context-Aware Responses** - Different response styles for different document types (textbook, resume, medical notes, etc.)
- 💬 **Chat History** - Keep track of all your questions and answers in the sidebar
- 🎙️ **Text-to-Speech** - Convert answers to audio (gTTS integration)

## 🛠️ Technology Stack

| Technology         | Purpose                                               |
| ------------------ | ----------------------------------------------------- |
| **Streamlit**      | Web interface and user interaction                    |
| **LangChain**      | AI framework for building language model applications |
| **Google Gemini API** | Gemini 1.5 Flash model for answering questions (free tier) |
| **FAISS**          | Vector database for fast document similarity search   |
| **PyMuPDF (fitz)** | PDF text extraction                                   |
| **gTTS**           | Google Text-to-Speech conversion                      |

## 📋 Requirements

- Python 3.8 or higher
- Google (Gemini) API Key — free at [Google AI Studio](https://aistudio.google.com/apikey)
- A working internet connection

## 🚀 Installation & Setup

### Step 1: Clone or Download the Project

```bash
cd gemini-pdf-study-assistant
```

### Step 2: Create a Virtual Environment

Creating a virtual environment keeps your project dependencies isolated from your system Python.

```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Your Google (Gemini) API Key (Free)

Get a free API key at [Google AI Studio](https://aistudio.google.com/apikey).

#### Option A: Using Environment Variables (Recommended)

```bash
# On Windows (PowerShell)
$env:GOOGLE_API_KEY = "your-api-key-here"

# On Windows (Command Prompt)
set GOOGLE_API_KEY=your-api-key-here

# On macOS/Linux
export GOOGLE_API_KEY="your-api-key-here"
```

#### Option B: Using .env File

1. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open the `.env` file and add your Google API Key:
   ```
   GOOGLE_API_KEY=your-actual-api-key-here
   ```

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📖 How to Use

1. **Upload a PDF**: Click "Upload your study material" and select a PDF file
2. **Choose Document Type**: Select the type of document you're working with (Textbook, Resume, Medical Notes, Other)
3. **Ask Questions**: Type your question in the text box
4. **Get Answers**:
   - Click "Ask Normally" for a standard answer
   - Click "Summarize Document" to get a quick overview
5. **View History**: Check the sidebar to see all previous questions and answers

## 🔧 Project Structure

```
gemini-pdf-study-assistant/
├── app.py                 # Main application code
├── requirements.txt       # Python dependencies
├── .env.example          # Example environment variables
├── README.md             # This file
├── DOCUMENTATION.md      # Detailed technical documentation
└── .venv/                # Virtual environment (created during setup)
```

## 💡 Key Concepts Explained

### What is FAISS?

FAISS (Facebook AI Similarity Search) is a vector database that efficiently stores and searches text embeddings. It allows the AI to quickly find relevant sections of your document related to your question.

### What are Embeddings?

Embeddings are mathematical representations of text that capture meaning. Google's Gemini embedding model converts text into vectors that can be compared to find similar content.

### What is LangChain?

LangChain is a framework that simplifies building applications with language models. In this project, it handles tasks like text splitting, embedding creation, vector search, and retrieval-enhanced prompting.

### What is Retrieval-Enhanced Q&A?

This app uses a lightweight retrieval flow with generation. The system:

1. Searches your document for relevant information
2. Provides that context to the AI
3. Generates answers based on your document content

## 📌 Current Scope

- Uses Google Gemini API for both embeddings and answer generation
- Performs retrieval from uploaded PDF content using FAISS
- Supports question answering, simplified explanations, summaries, and quiz generation
- Focuses on single-session PDF analysis in a Streamlit interface

## 🔒 Security & Best Practices

- ✅ API keys are stored in environment variables (never hardcoded)
- ✅ Virtual environment keeps dependencies isolated
- ✅ The `.env` file is not tracked in version control (add to `.gitignore`)
- ✅ Use a `.env.example` to show required environment variables

## 🐛 Troubleshooting

### Error: "Google (Gemini) API Key missing!"

**Solution**: Make sure you've set the `GOOGLE_API_KEY` environment variable. See the setup instructions above.

### Error: "Module not found"

**Solution**: Ensure you've activated the virtual environment and run `pip install -r requirements.txt`

### Application loads but gives API errors

**Solution**: Check if your Google API key is valid. Get or manage your key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### PDF text extraction issues

**Solution**: Some PDFs may have encoding issues. Try uploading a different PDF or ensure the PDF is not password-protected.

## 📚 Learning Path for Beginners

1. **Understand the basics**: Read [DOCUMENTATION.md](DOCUMENTATION.md) for detailed explanations
2. **Run the app**: Follow the installation steps and try it out
3. **Experiment**: Try different PDFs and document types
4. **Modify**: Try changing prompts or document types in the code
5. **Explore**: Look into LangChain and Google Gemini documentation for advanced features

## 🤝 Contributing

Feel free to fork this project and submit pull requests with improvements!

## 📄 License

This project is open source and available for educational purposes.

## 📞 Support & Resources

- **Google AI Studio (Gemini)**: https://aistudio.google.com/
- **Gemini API Docs**: https://ai.google.dev/docs
- **LangChain Docs**: https://python.langchain.com/
- **Streamlit Docs**: https://docs.streamlit.io/
- **FAISS Documentation**: https://faiss.ai/

---

**Happy Learning!** 🚀 If you have questions, check the DOCUMENTATION.md file for more detailed explanations.
