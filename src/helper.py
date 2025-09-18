import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# -------- Extract text from PDFs --------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


# -------- Split text into chunks --------
def get_text_chunks(text, chunk_size=1000, chunk_overlap=150):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)


# -------- Convert chunks into FAISS vector DB --------
def get_vector_stores(chunks, persist_path="./faiss_index"):
    docs = [Document(page_content=chunk) for chunk in chunks]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(persist_path):
        try:
            print("Loading existing FAISS index...")
            vs = FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
            return vs
        except:
            print("Failed to load existing index. Rebuilding...")

    print("Building FAISS index from chunks...")
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(persist_path)
    print("Saved FAISS index at", persist_path)
    return vs


# -------- Load TinyLlama LLM --------
def load_tinyllama(snapshot_folder):
    tokenizer = AutoTokenizer.from_pretrained(snapshot_folder, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot_folder,
        local_files_only=True,
        device_map=None  # CPU (use "auto" if GPU available)
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        truncation=True
    )

    llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={"temperature": 0.7, "max_length": 200})
    print("✅ TinyLlama loaded successfully!")
    return llm


# -------- Conversational Chain --------
def get_conversational_chain(vector_store, llm):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )
    return conversation_chain

