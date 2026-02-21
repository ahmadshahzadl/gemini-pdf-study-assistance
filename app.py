import streamlit as st
import os
from io import BytesIO
from gtts import gTTS
import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()  # Load GOOGLE_API_KEY from .env file

# IMPORTANT NOTE:
# To run this code, please set your Google (Gemini) API Key in a .env file
# or export it as an environment variable. Get a free key at https://aistudio.google.com/apikey
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Google (Gemini) API Key missing! Please set GOOGLE_API_KEY in environment variables.")
    st.stop()

# Free Gemini model for chat and generation
GEMINI_MODEL = "gemini-2.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"

# Extract text from uploaded PDF
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text



# Split text and create vectorstore
def create_vectorstore_from_text(text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.create_documents([text])

    # SECURE: Key is retrieved from environment variable
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key
    )
    return FAISS.from_documents(docs, embeddings)

# Select Document Type
doc_type = st.selectbox("Document Type", ["Textbook", "Resume", "Medical Notes", "Other"])

# Choose prompt style based on document type
if doc_type == "Resume":
    prompt_template = "You are an HR expert. Use the context to answer questions like a resume reviewer.\n\nContext: {context}\nQuestion: {question}\nAnswer:"
elif doc_type == "Textbook":
    prompt_template = "Use the following context to answer like you're explaining to a 15-year-old.\n\nContext: {context}\nQuestion: {question}\nAnswer:"
elif doc_type == "Medical Notes":
    prompt_template = "Use the following context to give clear and accurate medical explanations.\n\nContext: {context}\nQuestion: {question}\nAnswer:"
else:
    prompt_template = "Use the context to answer the question clearly and concisely.\n\nContext: {context}\nQuestion: {question}\nAnswer:"

SIMPLE_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Streamlit App Configuration
st.set_page_config(page_title="Gemini PDF Study Assistant", layout="wide")
st.title("Gemini PDF Study Assistant")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload your study material (PDF only)", type=["pdf"])
query = st.text_input("Ask your question here:")

# Sidebar for History
with st.sidebar:
    st.markdown("## Chat History")
    if st.session_state.chat_history:
        for sender, msg in reversed(st.session_state.chat_history):
            st.markdown(f"**{sender}:** {msg}")
    else:
        st.info("No history yet.")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        vectorstore = create_vectorstore_from_text(text)

        # Document Summary Button
        if st.button("Summarize Document"):
            summary_prompt = "Summarize this document in 5 concise bullet points:\n\n" + text[:1500]
            llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=api_key)
            summary = llm.invoke(summary_prompt)
            st.markdown("### Document Summary:")
            st.write(summary.content if hasattr(summary, "content") else summary)

    if query:
        col1, col2 = st.columns(2)

        # Standard Q&A Column
        with col1:
            if st.button("Ask Normally"):
                retriever = vectorstore.as_retriever()
                docs = retriever.invoke(query)
                context = "\n\n".join(doc.page_content for doc in docs)
                prompt = SIMPLE_PROMPT.format(context=context, question=query)
                llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=api_key)
                response = llm.invoke(prompt)
                result = response.content if hasattr(response, "content") else str(response)
                st.markdown("**Answer:**")
                st.write(result)

                # Text-to-Speech
                tts = gTTS(text=str(result))
                audio_bytes = BytesIO()
                tts.write_to_fp(audio_bytes)
                audio_bytes.seek(0)
                st.audio(audio_bytes, format='audio/mp3')

                # Update History
                st.session_state.chat_history.append(("You", query))
                st.session_state.chat_history.append(("AI", result))

        # Simplified Explanation Column
        with col2:
            if st.button("Explain Like I'm 15"):
                docs_and_scores = vectorstore.similarity_search_with_score(query, k=2)
                # Filter documents based on similarity score
                relevant_docs = [doc for doc, score in docs_and_scores if score < 0.7]

                if not relevant_docs:
                    st.error("Sorry, topic not found in the document.")
                else:
                    context = "\n\n".join(doc.page_content for doc in relevant_docs)
                    prompt = SIMPLE_PROMPT.format(context=context, question=query)
                    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=api_key)
                    resp = llm.invoke(prompt)
                    response = resp.content if hasattr(resp, "content") else str(resp)

                    st.markdown("**Simplified Answer:**")
                    st.write(response)

                    # Text-to-Speech
                    tts = gTTS(text=str(response))
                    audio_bytes = BytesIO()
                    tts.write_to_fp(audio_bytes)
                    audio_bytes.seek(0)
                    st.audio(audio_bytes, format='audio/mp3')

                    # Update History
                    st.session_state.chat_history.append(("You", query))
                    st.session_state.chat_history.append(("AI (Simple)", response))

    # Quiz Generation
    if st.button("Generate Quiz"):
        prompt = "Generate 3 MCQs from the following study notes:\n\n" + text[:1000]
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=api_key)
        response = llm.invoke(prompt)
        st.markdown("### Quiz Questions:")
        st.write(response.content if hasattr(response, "content") else response)
