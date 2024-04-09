from flask import Flask, render_template, request, jsonify
from deepgram import DeepgramClient
from deepgram import PrerecordedOptions, FileSource
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from fpdf import FPDF
from utils import save_conversations_to_pdf
from utils import format_conversations
import os

app = Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

API_KEY = "ae1218dfff62d19fdd9315dfe79dc1e7a243d1f6"
deepgram = DeepgramClient(API_KEY)
OPENAI_API_KEY = "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"
embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
llm_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=llm_name, temperature=0, api_key=OPENAI_API_KEY)

@app.route('/analyze_audio', methods=['POST'])
def analyze_audio():
    # Check if an audio file is included in the request
    # print(request)
    if 'file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['file']
    # print(audio_file)

    # Save the audio file temporarily
    audio_file_path = "temp_audio.mp3"
    audio_file.save(audio_file_path)
    if os.path.exists(audio_file_path):
        print("exists")
    print("started Try")
    print(audio_file)

    try:
        # Transcribe the audio file using Deepgram
        with open(audio_file_path, "rb") as file:
            buffer_data = file.read()
        # print(buffer_data)
        payload: FileSource = {"buffer": buffer_data,}
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=False,
            dictation=True,
            punctuate=True,
            diarize=True,
            utterances=True,
            filler_words=False
        )
        print("After buffer data")
        text_output = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
        # transcript = text_output.results["channels"][0]["alternatives"][0]["transcript"]
        print("Before format")
        # Format conversations
        conversations = format_conversations(text_output)

        # Save conversations to a PDF text file
        save_conversations_to_pdf(conversations, "conversations.pdf")
        print("After conversation")
        # Split text into chunks for processing
        loader = PyPDFLoader("conversations.pdf")
        pages = loader.load()
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=150,
            length_function=len
        )
        docs = text_splitter.split_documents(pages)

        # Generate embeddings vector database
        persist_directory = 'chroma/'
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory=persist_directory
        )

        # Build prompt for sentiment analysis
        template = """Use the following pieces of context to answer the question at the end. Keep the answer as concise as possible and don't summarize the content as output. You are a sentiment analysis assistant for a conversation.
        {context}
        Question: {question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        # Run sentiment analysis chain
        question = "Define the sentiment of each speaker and provide me the feelings of each person. For example: '[Speaker_0] likes a sport. It seems he cares about his health.' Here 'cares about health' is the sentiment, '[Speaker_1] pretends to be smart.' Here 'pretending' is the sentiment?"
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        result = qa_chain({"query": question})

        # Return the sentiment analysis result
        return jsonify({'result': "sentiment    "+result['result']}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
 
    finally:
        # Clean up temporary files
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)
