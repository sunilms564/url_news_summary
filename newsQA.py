import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Added import

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env (especially openai api key)

st.title("News Research Tool 📰")  # Added emoji for visual appeal
st.sidebar.title("News Article URLs")

urls = [st.sidebar.text_input(f"URL {i + 1}") for i in range(3)]

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    try:
        # Load data
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data Loading...")  # Added ellipsis for better feedback
        data = loader.load()

        # Split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitting...")  # Modified text for clarity
        docs = text_splitter.split_documents(data)

        # Create embeddings and save to FAISS index
        embeddings = OpenAIEmbeddings()
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vectors Building...")  # Modified text for clarity
        time.sleep(2)

        # Save the FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_openai, f)

        main_placeholder.text("Data Processing Completed! ✅")  # Simplified completion message
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

query = st.text_input("Ask a Question: ")  # Modified input prompt for clarity
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            # Display answer
            st.header("Answer")
            st.write(result["answer"])

            # Display sources
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for idx, source in enumerate(sources_list, start=1):
                    st.write(f"{idx}. {source}")
