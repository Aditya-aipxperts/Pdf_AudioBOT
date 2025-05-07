# Pdf_AudioBOT

Pdf_AudioBOT is an interactive chatbot that allows users to ask questions about the contents of a PDF document. It uses speech recognition for question input and text-to-speech for spoken responses. This project integrates various technologies, including OpenAI's GPT models, Whisper for audio transcription, ElevenLabs for speech synthesis, and Langchain for document processing.

## Features

- **PDF Text Extraction**: Extracts text from a PDF document using `pypdf`.
- **Text Embedding**: Embeds the extracted text using the Sentence Transformers model and stores it in a FAISS index for efficient retrieval.
- **Speech Recognition**: Records audio input from the user and converts it into text using the Whisper model.
- **LLM-based Question Answering**: Leverages OpenAI's GPT-based models to answer questions based on the content of the PDF.
- **Text-to-Speech**: Converts the chatbot's responses into speech using ElevenLabs API or falls back to Google TTS (gTTS).
- **Environment Variables for API Keys**: All sensitive API keys (for OpenAI and ElevenLabs) are stored in `.env` files to keep them secure.