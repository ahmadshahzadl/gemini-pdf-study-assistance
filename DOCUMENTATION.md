# Gemini PDF Study Assistant - Complete Technical Documentation

This document explains how the Gemini PDF Study Assistant application works, breaking down complex concepts into beginner-friendly explanations.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Installation & Environment Setup](#installation--environment-setup)
3. [Code Breakdown](#code-breakdown)
4. [How It All Works Together](#how-it-all-works-together)
5. [Understanding Key Technologies](#understanding-key-technologies)
6. [Workflow Diagrams](#workflow-diagrams)
7. [Advanced Features](#advanced-features)
8. [Customization Guide](#customization-guide)
9. [Known Limitations](#known-limitations)

---

## Architecture Overview

The Gemini PDF Study Assistant uses a **retrieval-enhanced Q&A** architecture. Here's what that means:

```
User Uploads PDF
    ↓
PDF Text Extraction
    ↓
Text Splitting (Chunking)
    ↓
Embedding Generation (Converting text to vectors)
    ↓
Storing in FAISS Vector Database
    ↓
User Asks Question
    ↓
Convert Question to Embedding
    ↓
Search FAISS for Similar Content
    ↓
Pass Retrieved Content + Question to Gemini
    ↓
Gemini Generates Answer
    ↓
Display to User
```

### Why retrieval + generation?

Without retrieval, the AI would only know what it learned during training (which is old and limited). With retrieval + generation, the AI:

- Reads YOUR specific documents
- Finds relevant sections for YOUR question
- Generates answers based on YOUR content

This ensures answers are accurate and specific to your documents.

---

## Installation & Environment Setup

### What is a Virtual Environment?

A virtual environment is an isolated Python installation. Think of it like having a separate copy of Python just for this project, so your project's dependencies don't conflict with other projects.

### Step-by-Step Setup

#### 1. Create Virtual Environment

```bash
python -m venv .venv
```

**What this does:**

- Creates a folder `.venv` with a fresh Python installation
- This folder is listed in `.gitignore` so it's not tracked in version control

#### 2. Activate Virtual Environment

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
.venv\Scripts\activate.bat
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

**How to know it's activated:**
Your terminal will show `(.venv)` at the beginning of the prompt.

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**What `requirements.txt` contains:**

```
streamlit                 # Web framework for building the interface
langchain-google-genai    # LangChain integration with Google Gemini
langchain-community      # Additional LangChain utilities
langchain-core           # LangChain core
pymupdf                  # PDF reading library (fitz)
gTTS                     # Google Text-to-Speech
faiss-cpu                # Vector database (CPU version)
python-dotenv            # Load environment variables from .env file
```

#### 4. Set Google (Gemini) API Key (Free at https://aistudio.google.com/apikey)

**Method 1: Direct Environment Variable (Windows PowerShell)**

```powershell
$env:GOOGLE_API_KEY = "your-gemini-api-key-here"
```

**Method 2: Using .env File (Recommended for beginners)**

Create a `.env` file:

```
GOOGLE_API_KEY=your-gemini-api-key-here
```

Then in Python, load it:

```python
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
```

**Why use .env?**

- Keeps secrets out of code
- Easy to switch between test/production keys
- Never commit `.env` to version control

---

## Known Limitations

- The app currently expects one uploaded PDF per active processing flow
- It relies on external Gemini API availability and quota limits
- Retrieved context is chunk-based and may miss long-range dependencies in complex documents
- No persistent database is used; processing is session-scoped in Streamlit

---

## Code Breakdown

### Main Application Structure

#### 1. Imports & Setup

```python
import streamlit as st
import os
from io import BytesIO
from gtts import gTTS
import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
```

**What each import does:**

| Import                         | Purpose                               |
| ------------------------------ | ------------------------------------- |
| `streamlit`                    | Creates the web interface             |
| `os`                           | Accesses environment variables        |
| `BytesIO`                      | Handles file streams in memory        |
| `gTTS`                         | Converts text to speech               |
| `fitz` (PyMuPDF)               | Reads PDF files and extracts text     |
| `FAISS`                        | Vector database for similarity search |
| `GoogleGenerativeAIEmbeddings` | Converts text to mathematical vectors |
| `ChatGoogleGenerativeAI`       | Access to Gemini language model       |

#### 2. API Key Validation

```python
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Google (Gemini) API Key missing! Please set GOOGLE_API_KEY in environment variables.")
    st.stop()
```

**Why this matters:**

- Checks if API key exists before proceeding
- Prevents cryptic errors later
- User-friendly error message
- `st.stop()` halts the app execution

#### 3. PDF Text Extraction Function

```python
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text
```

**How it works:**

```
Input: PDF file from upload
    ↓
fitz.open(): Opens the PDF in memory
    ↓
Loop through each page
    ↓
page.get_text(): Extracts text from page
    ↓
Concatenate all text
    ↓
Return: Full document text as string
```

**Why `BytesIO`?**

- Stores the file in memory instead of disk
- Faster and more efficient
- Doesn't clutter the server

#### 4. Vector Store Creation Function

```python
def create_vectorstore_from_text(text):
    from langchain.text_splitter import CharacterTextSplitter
    splitter = CharacterTextSplitter(
        chunk_size=500,      # Size of each chunk
        chunk_overlap=100    # Overlap between chunks
    )
    docs = splitter.create_documents([text])

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    return FAISS.from_documents(docs, embeddings)
```

**Step-by-step explanation:**

**Step 1: Text Splitting**

```
Original Text: "The photosynthesis process is... [10,000 words]"

After splitting into 500-character chunks with 100-char overlap:

Chunk 1: [0:500 chars]
Chunk 2: [400:900 chars]    ← 100 chars overlap with Chunk 1
Chunk 3: [800:1300 chars]   ← 100 chars overlap with Chunk 2
```

**Why split?**

- Gemini has token limits
- Allows searching for relevant sections
- Better performance than processing entire document

**Step 2: Creating Embeddings**

```
Chunk: "Photosynthesis is the process..."
    ↓
Google Gemini API converts to vector
    ↓
Vector: [0.123, -0.456, 0.789, ...]  (768 dimensions)
```

**What are embeddings?**

- Mathematical representation of text meaning
- Similar texts have similar vectors
- Allows us to measure text similarity

**Step 3: Storing in FAISS**

```
Chunk 1 → Embedding 1 ┐
Chunk 2 → Embedding 2 ├→ Stored in FAISS
Chunk 3 → Embedding 3 ┘
```

FAISS creates an index that allows fast searching.

#### 5. Document Type Selection

```python
doc_type = st.selectbox("Document Type",
    ["Textbook", "Resume", "Medical Notes", "Other"])
```

**Why different types?**

Different documents need different response styles:

```python
if doc_type == "Resume":
    prompt_template = "You are an HR expert. Use the context to answer..."
elif doc_type == "Textbook":
    prompt_template = "Use the following context to answer like you're explaining to a 15-year-old..."
elif doc_type == "Medical Notes":
    prompt_template = "Use the following context to give clear and accurate medical explanations..."
```

**Prompt Engineering Concept:**

By changing the prompt, we change how the AI behaves:

- Resume → Acts like HR professional
- Textbook → Explains simply, like teaching
- Medical → Uses precise medical language

This is called "prompt engineering" - crafting instructions to get better results.

#### 6. Main Application Interface

```python
st.set_page_config(page_title="Gemini PDF Study Assistant", layout="wide")
st.title("Gemini PDF Study Assistant")
```

**What these do:**

- `set_page_config()`: Sets browser tab title and layout
- `title()`: Displays main heading in the app

#### 7. Session State Management

```python
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
```

**What is session state?**

In Streamlit, the script re-runs every time user interacts with the app. Session state preserves data across re-runs.

```
First Load: chat_history = []
User Asks Question: script re-runs, chat_history still = []
Add to history: chat_history = [("User", "Q1")]
User Asks Another: script re-runs, chat_history = [("User", "Q1")]
```

**Without session state:**

- Chat history would reset every interaction
- Same question answered repeatedly
- Bad user experience

#### 8. File Upload & Processing

```python
uploaded_file = st.file_uploader("Upload your study material (PDF only)",
                                 type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        vectorstore = create_vectorstore_from_text(text)
```

**What happens:**

1. User uploads PDF
2. `st.spinner()` shows "Processing PDF..." while working
3. Extract text from PDF
4. Create vector database from extracted text

#### 9. Document Summarization

```python
if st.button("Summarize Document"):
    summary_prompt = "Summarize this document in 5 concise bullet points:\n\n" + text[:1500]
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    summary = llm.invoke(summary_prompt)
    st.write(summary.content if hasattr(summary, "content") else summary)
```

**How it works:**

1. Create a prompt asking for 5-point summary
2. Use `text[:1500]` - only first 1500 characters (saves tokens & cost)
3. Send to ChatGoogleGenerativeAI (Gemini 1.5 Flash)
4. Display the response

#### 10. Question Answering

```python
if st.button("Ask Normally"):
    qa = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key),
        retriever=vectorstore.as_retriever()
    )
    result = qa.run(query)
```

**The RAG Pipeline:**

```
User Question: "What is photosynthesis?"
    ↓
Convert to embedding: [0.234, -0.567, ...]
    ↓
FAISS searches: "Find top 4 similar chunks"
    ↓
Retrieved Chunks:
    - "Photosynthesis is the process of..."
    - "During photosynthesis, plants..."
    - "The chlorophyll absorbs light..."
    - "This process produces oxygen..."
    ↓
Create context: Combine retrieved chunks
    ↓
Send to Gemini:
    Context: [All retrieved text]
    Question: "What is photosynthesis?"
    ↓
Gemini generates answer based on context
    ↓
Return: "Photosynthesis is the process where plants..."
```

---

## How It All Works Together

### Complete User Journey

```
1. USER LOADS APP
   ↓
2. APP CHECKS API KEY
   └─ If missing → Show error and stop
   └─ If present → Continue
   ↓
3. USER UPLOADS PDF
   ↓
4. APP PROCESSES PDF
   ├─ Extract text using PyMuPDF
   ├─ Split text into chunks
   ├─ Convert chunks to embeddings (Google Gemini)
   └─ Store in FAISS database
   ↓
5. USER ASKS QUESTION
   ↓
6. APP FINDS RELEVANT SECTIONS
   ├─ Convert question to embedding
   ├─ Search FAISS for similar chunks
   └─ Retrieve top matching sections
   ↓
7. APP GENERATES ANSWER
   ├─ Pass context + question to Gemini
   └─ Gemini generates tailored response
   ↓
8. APP DISPLAYS ANSWER
   ├─ Show response in chat
   ├─ Add to chat history
   └─ Store for future reference
   ↓
9. USER ASKS ANOTHER QUESTION
   └─ Repeat from step 5
```

---

## Understanding Key Technologies

### 1. Streamlit

**What is it?**
Framework for building web apps with Python (no HTML/CSS/JavaScript needed).

**Key Streamlit Components Used:**

```python
st.title()           # Large heading
st.selectbox()       # Dropdown menu
st.file_uploader()   # File upload button
st.text_input()      # Text input field
st.button()          # Clickable button
st.spinner()         # Loading indicator
st.write()           # Display text/data
st.columns()         # Create side-by-side layout
st.sidebar           # Sidebar content
st.session_state     # Persistent data storage
```

**Why Streamlit?**

- Pure Python (no web dev needed)
- Fast prototyping
- Built-in widgets
- Hot reload (updates on file save)
- Automatic re-running

### 2. LangChain

**What is it?**
Framework for building applications with language models.

**Core Components:**

| Component           | Purpose                                         |
| ------------------- | ----------------------------------------------- |
| **LLM**             | Language model wrapper (ChatGoogleGenerativeAI) |
| **Retriever**       | Searches vector store for relevant documents    |
| **Chain**           | Connects components together                    |
| **Prompt Template** | Template for formatting prompts                 |
| **Document**        | Represents a chunk of text with metadata        |

**The RetrievalQA Chain:**

```python
RetrievalQA = Retriever + LLM Chain
           = (Find docs) + (Generate answer)
```

How it works:

1. **Retriever** gets relevant documents from vector store
2. **Combine** documents into context
3. **LLM** generates answer using context + question
4. **Return** final answer

### 3. Google Gemini API (Free Tier)

**What models are used?**

| Model                  | Use Case                   | Cost      | Speed |
| ---------------------- | -------------------------- | --------- | ----- |
| `models/embedding-001` | Converting text to vectors | Free tier | Fast  |
| `gemini-1.5-flash`     | Answering questions        | Free tier | Fast  |

**Pricing:** Free tier available via [Google AI Studio](https://aistudio.google.com/apikey). Rate limits apply; see Google’s current limits.

### 4. FAISS (Vector Database)

**What is it?**
Fast similarity search engine for vectors.

**Why not just search text directly?**

❌ Bad approach:

```python
for chunk in chunks:
    if "photosynthesis" in chunk:
        return chunk
```

- Only finds exact matches
- Misses related concepts
- Slow for large documents

✅ Good approach (FAISS):

```python
similar_chunks = vectorstore.similarity_search(question_embedding, k=4)
```

- Finds semantic matches
- Catches related concepts
- Fast even with millions of chunks

**How FAISS works:**

```
1. Calculate similarity between question and all chunk embeddings
2. Return top k most similar chunks
3. Uses clever indexing to do this very fast

Distance between vectors:
- Similar texts → Small distance
- Different texts → Large distance

Question: "What is photosynthesis?"
Embedding: [0.234, -0.567, 0.890]

Chunk 1: "Photosynthesis is..." → Distance: 0.05 ← VERY SIMILAR
Chunk 2: "What color is..." → Distance: 0.92 ← DISSIMILAR
Chunk 3: "Plants use photosynthesis..." → Distance: 0.12 ← SIMILAR
```

### 5. PyMuPDF (fitz)

**What is it?**
Python library for reading PDF files.

**What it does:**

```python
import fitz

# Open PDF
doc = fitz.open("document.pdf")

# Get page count
print(doc.page_count)  # Output: 50

# Extract text from page
page = doc[0]  # First page
text = page.get_text()

# Close document
doc.close()
```

**Supported Operations:**

- Extract text
- Get page count
- Extract images
- Handle annotations
- Read metadata

---

## Workflow Diagrams

### Initialization Workflow

```
App Start
├─ Load imports
├─ Read API key from environment
├─ Validate API key exists
│  ├─ If missing → Error + Stop
│  └─ If present → Continue
├─ Initialize Streamlit UI
├─ Create session state (chat_history)
└─ Ready for user input
```

### PDF Processing Workflow

```
User Uploads PDF
├─ Validate file type (must be .pdf)
├─ Read file into memory (BytesIO)
├─ Open PDF with PyMuPDF
│  ├─ Iterate through pages
│  ├─ Extract text from each page
│  └─ Concatenate all text
├─ Split text into chunks
│  ├─ Chunk size: 500 characters
│  ├─ Overlap: 100 characters
│  └─ Create LangChain Document objects
├─ Generate embeddings
│  ├─ Call Google Gemini Embedding API
│  ├─ Get vector (768 dimensions) for each chunk
│  └─ Store in memory
├─ Create FAISS vector store
│  ├─ Index all embeddings
│  └─ Enable fast similarity search
└─ Ready to answer questions
```

### Question Answering Workflow

```
User Asks Question
├─ Validate question not empty
├─ Create embedding of question
│  └─ Call Google Gemini Embedding API
├─ Search FAISS vector store
│  ├─ Find top 4 most similar chunks
│  ├─ Calculate similarity distances
│  └─ Return matched chunks
├─ Create context from retrieved chunks
│  ├─ Combine chunk text
│  └─ Format with prompt template
├─ Call ChatGoogleGenerativeAI (Gemini 1.5 Flash)
│  ├─ System message: Document type specific instruction
│  ├─ User message: Context + Question
│  └─ Stream response from API
├─ Add to chat history
│  ├─ Store user question
│  ├─ Store AI answer
│  └─ Display in sidebar
└─ Display answer to user
```

---

## Advanced Features

### 1. Multiple Document Types with Prompt Engineering

The app handles different document types by using specialized prompts:

```python
prompts = {
    "Resume": "You are an HR expert analyzing this resume...",
    "Textbook": "Explain this like you're teaching a 15-year-old...",
    "Medical": "Provide accurate medical information...",
}
```

**Why this matters:**

- Same document, different response styles
- Optimizes for use case
- Shows power of prompt engineering

### 2. Chat History in Sidebar

```python
with st.sidebar:
    st.markdown("## Chat History")
    for sender, msg in reversed(st.session_state.chat_history):
        st.markdown(f"**{sender}:** {msg}")
```

**Features:**

- Shows all previous Q&A
- Reversed order (newest first)
- Persists during session

### 3. Document Summarization

```python
text[:1500]  # Only first 1500 chars
```

**Why limit?**

- Saves API tokens (costs less)
- Still captures main ideas
- Faster response

### 4. Two-Column Layout

```python
col1, col2 = st.columns(2)

with col1:
    # Ask Normally button and response

with col2:
    # Other features could go here
```

**Benefits:**

- Better use of screen space
- Side-by-side comparison possible
- Organized interface

---

## Customization Guide

### How to Modify Document Types

Current document types in `app.py`:

```python
doc_type = st.selectbox("Document Type",
    ["Textbook", "Resume", "Medical Notes", "Other"])
```

**To add a new type:**

1. Add to selectbox:

```python
doc_type = st.selectbox("Document Type",
    ["Textbook", "Resume", "Medical Notes", "Code Review", "Other"])
```

2. Add corresponding prompt:

```python
elif doc_type == "Code Review":
    prompt_template = "You are an expert code reviewer. Analyze the provided code and answer questions about it.\n\nContext: {context}\nQuestion: {question}\nAnswer:"
```

### How to Change Text Chunk Size

Current settings:

```python
splitter = CharacterTextSplitter(
    chunk_size=500,      # Change this
    chunk_overlap=100    # Or this
)
```

**Effect of changing:**

| Setting          | Chunk Size           | Effect                                   |
| ---------------- | -------------------- | ---------------------------------------- |
| Increase to 1000 | Fewer, larger chunks | Better context but less precise search   |
| Decrease to 250  | More, smaller chunks | More precise but might miss context      |
| Increase overlap | More overlap         | Better context continuity but more data  |
| Decrease overlap | Less overlap         | Faster processing but might lose context |

**Recommended values:**

- `chunk_size=500, overlap=100` for books/notes
- `chunk_size=250, overlap=50` for technical documents
- `chunk_size=1000, overlap=200` for long-form articles

### How to Use Different Language Models

Current model (free):

```python
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
```

**To use Gemini 1.5 Pro (more capable, still free tier):**

```python
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key)
```

**Model comparison:**

| Model              | Speed  | Quality   |
| ------------------ | ------ | --------- |
| `gemini-1.5-flash` | Fast   | Good      |
| `gemini-1.5-pro`   | Slower | Excellent |

### How to Add Text-to-Speech

The code has gTTS import:

```python
from gtts import gTTS
```

**To enable audio output:**

```python
# After getting the answer
answer_text = result  # This is the answer

# Generate speech
tts = gTTS(text=answer_text, lang='en')
audio = BytesIO()
tts.write_to_fp(audio)
audio.seek(0)

# Play in Streamlit
st.audio(audio, format='audio/mp3')
```

---

## Common Questions & Answers

### Q: Why is my API call slow?

**A:** Gemini API can take 1-5 seconds. Use `st.spinner()` to show loading message.

### Q: Why is the summary so short?

**A:** It only uses `text[:1500]` (first 1500 characters). Change to `text[:3000]` for longer summary.

### Q: Can I use this with other PDF formats?

**A:** PyMuPDF supports: PDF, XPS, EPUB, CBZ, etc. Change `type=["pdf"]` to `type=["pdf", "epub"]`

### Q: How do I reduce API costs?

**A:**

- Smaller chunk sizes
- Fewer question answers
- Use gemini-1.5-flash instead of gemini-1.5-pro
- Retrieve fewer chunks (change `k=4` to `k=2`)

### Q: Can I save results to database?

**A:** Yes! Add SQLite or MongoDB to store chats:

```python
import sqlite3

conn = sqlite3.connect('chats.db')
cursor = conn.cursor()
cursor.execute('INSERT INTO chats VALUES (?, ?)', (question, answer))
conn.commit()
```

---

## Summary

AI Study Buddy combines:

1. **PDF Processing** (PyMuPDF) - Extract content
2. **Vector Embeddings** (Google Gemini) - Convert to searchable format
3. **Vector Database** (FAISS) - Fast semantic search
4. **LLM** (Gemini 1.5 Flash) - Generate intelligent answers
5. **Web Framework** (Streamlit) - User-friendly interface

This creates an intelligent assistant that understands your documents and answers questions accurately.

---

**For more information, refer to:**

- [README.md](README.md) - Setup instructions
- Google AI Studio: https://aistudio.google.com/
- Gemini API Docs: https://ai.google.dev/docs
- LangChain Docs: https://python.langchain.com/
- Streamlit Docs: https://docs.streamlit.io/
