import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

from pytube import YouTube
from pathlib import Path
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

import requests

API_URL = "https://api-inference.huggingface.co/models/openai/whisper-tiny"
headers = {"Authorization": os.getenv("HUGGINGFACE_API_KEY")}


# PyTube function for YouTube video
def save_audio(url):
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=True).first()
    out_file = video.download()
    base, ext = os.path.splitext(out_file)
    file_name = base + '.mp3'
    try:
        os.rename(out_file, file_name)
    except WindowsError:
        os.remove(file_name)
        os.rename(out_file, file_name)
    audio_filename = Path(file_name).stem+'.mp3'
    return audio_filename

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question based on the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context"\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro")

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




st.set_page_config(layout="wide", page_title="ChatVideo", page_icon="🔊")

st.title("Chat with Your Youtube Video using LLM")

input_source = st.text_input("Enter the YouTube video URL")

if input_source:
    col1, col2 = st.columns(2)

    with col1:
        st.info("Your uploaded video")
        st.video(input_source)
        audio_filename = save_audio(input_source)
        transription = query(audio_filename)
        st.subheader('Video Transription')
        st.info(transription['text'])
        raw_text = transription['text']
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        st.success("Done")
    with col2:
        st.info("Chat Below")
        user_question = st.chat_input("Ask your Query here...")
        
        if user_question:
            with st.chat_message("user"):
                st.write(user_question)
            response = user_input(user_question)
            with st.chat_message("assistant"):
                st.write(str(response['output_text']))

        
