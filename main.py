import os
import pypdf
import sounddevice as sd
from dotenv import load_dotenv
from scipy.io.wavfile import write
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from elevenlabs import generate, play, set_api_key
from faster_whisper import WhisperModel
from gtts import gTTS
from openai import OpenAI


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")

set_api_key(ELEVEN_API_KEY)

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)
LLM_MODEL = "llama3-70b-8192"  # Pick one available in Groq

# PDF path
pdf_file_path = "pdfs/notes.pdf"

# Load PDF and extract text 
def load_pdf(file_path):
    reader = pypdf.PdfReader(open(file_path, "rb"))
    text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

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
    print("🎙️ Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    
    write(filename, fs, audio)
    print("✅ Audio saved:", filename)

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

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# Speak response using ElevenLabs or gTTS
def speak_response(text):
    try:
        model_id = "GTiuKhCAJGILEG2FelGh" # "goT3UYdM9bhm0n2lmKQx" #GTiuKhCAJGILEG2FelGh  #MF4J4IDTRo0AxOO4dpFR
        audio = generate(text=text, voice=model_id, model="eleven_multilingual_v1")
        print("elevenlabs audio generated")
        play(audio)
    except Exception as e:
        print(f"Error generating speech: {e}")
        print("Falling back to gTTS...")
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("response.mp3")
        # os.system("afplay response.mp3")
        os.system("start response.mp3")

# MAIN FLOW 
def run_chatbot():
    print("🔹 Loading PDF...")
    pdf_text = load_pdf(pdf_file_path)

    print("🔹 Creating vector index...")
    db = embed_text(pdf_text)

    print("🎙️ Speak your question (5 seconds)...")
    record_audio()

    query = transcribe_audio("audio/question.wav")
    print("🗣️ You asked:", query)

    print("🧠 Thinking...")
    answer = ask_question(query, db)
    print("🤖 Bot says:", answer)

    print("🔊 Speaking...")
    speak_response(answer)

if __name__ == "__main__":
    run_chatbot()
