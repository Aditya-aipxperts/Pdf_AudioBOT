import os
import pypdf
import sounddevice as sd
from env import load_dotenv
from scipy.io.wavfile import write
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from elevenlabs import generate, play, set_api_key
from faster_whisper import WhisperModel
from gtts import gTTS
from openai import OpenAI
import streamlit as st

load_dotenv()

GROQ_API_KEY = "GROQ_API_KEY"
ELEVEN_API_KEY = "ELEVENLABS_API_KEY"
set_api_key(ELEVEN_API_KEY)

client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
LLM_MODEL = "llama3-70b-8192"

# PDF path
pdf_file_path = "pdfs/notes.pdf"

# Load PDF and extract text
def load_pdf(file):
    try:
        reader = pypdf.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        if not text:
            raise ValueError("No text found in PDF.")
        return text
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return ""

# Split and embed PDF content
def embed_text(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    return db

# Record from mic
def record_audio(duration=5, filename="audio/question.wav"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fs = 44100
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)

# Transcribe using Whisper
def transcribe_audio(audio_path):
    model = WhisperModel("base", compute_type="auto")
    segments, _ = model.transcribe(audio_path)
    return " ".join([seg.text for seg in segments])

# Ask LLM with context
def ask_question(query, db):
    docs = db.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"Answer the question based only on this PDF:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    response = client.chat.completions.create(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

# Speak response using ElevenLabs or gTTS
def speak_response(text):
    try:
        model_id = "GTiuKhCAJGILEG2FelGh"
        audio = generate(text=text, voice=model_id, model="eleven_multilingual_v1")
        play(audio)
    except Exception as e:
        print(f"Error generating speech: {e}")
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        os.system("start response.mp3")

# Streamlit App UI and Logic
st.set_page_config(page_title="🎙️ Audio Chatbot", layout="centered")
st.title("🎙️ Speak with your PDF Chatbot")

# Load PDF content and embed it into the vector database
pdf_text = load_pdf(pdf_file_path)
db = embed_text(pdf_text)

# Upload a new PDF if the user wants
uploaded_pdf = st.file_uploader("Upload a new PDF", type="pdf")
if uploaded_pdf:
    # Use the uploaded file as a byte stream
    pdf_text = load_pdf(uploaded_pdf)
    if pdf_text:
        db = embed_text(pdf_text)
        st.success("PDF uploaded and indexed!")

# Record and transcribe audio
if st.button("🎤 Record your question"):
    st.write("Recording your question (5 seconds)...")
    record_audio(duration=5, filename="audio/question.wav")
    st.write("Recording complete. Transcribing...")

    query = transcribe_audio("audio/question.wav")
    st.write(f"**Your Question:** {query}")

    # Ask the question to the chatbot
    st.write("🧠 Thinking...")
    answer = ask_question(query, db)
    st.write(f"**Answer:** {answer}")

    # Speak the answer aloud
    speak_response(answer)
    st.success("✅ Spoken response played.")

