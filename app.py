import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import TiDBVectorStore
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import create_engine, text
import pymysql

pymysql.install_as_MySQLdb()

# --- Configuration using Streamlit Secrets ---
# Streamlit will automatically read the secrets from your .streamlit/secrets.toml file
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

TIDB_CONNECTION_STRING = st.secrets["TIDB_CONNECTION_STRING"]

# --- Helper Functions ---

def get_pdf_text(pdf_docs):
    """Extracts text from a list of PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits text into smaller chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Creates and stores a vector store in TiDB Cloud."""
    if not TIDB_CONNECTION_STRING:
        st.error("TiDB connection string is not set. Please set the TIDB_CONNECTION_STRING environment variable.")
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        TiDBVectorStore.from_texts(
            texts=text_chunks,
            embedding=embeddings,
            connection_string=TIDB_CONNECTION_STRING,
            table_name="document_vectors"
        )
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def get_conversational_chain():
    """Creates a conversational chain for question answering."""
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    chain = load_qa_chain(model, chain_type="stuff")
    return chain

def user_input(user_question):
    """Handles user input and gets a response from the chatbot."""
    if not TIDB_CONNECTION_STRING:
        st.error("TiDB connection string is not set. Please set the TIDB_CONNECTION_STRING environment variable.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        vector_store = TiDBVectorStore(
            connection_string=TIDB_CONNECTION_STRING,
            embedding_function=embeddings,
            table_name="document_vectors"
        )
        docs = vector_store.similarity_search(user_question)
    except Exception as e:
        st.error(f"Error connecting to or querying the vector store: {e}")
        # Check if the table exists
        try:
            engine = create_engine(TIDB_CONNECTION_STRING)
            with engine.connect() as connection:
                connection.execute(text("SELECT 1 FROM document_vectors LIMIT 1"))
        except Exception as table_error:
            st.warning("The 'document_vectors' table might not exist yet. Please upload a document to create it.")
        return

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# --- Streamlit App ---

def main():
    st.set_page_config(page_title="Chat with Your Docs", layout="wide")
    st.header("Chat with Your Documents using Gemini and TiDB 💬")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Sidebar for uploading documents
    with st.sidebar:
        st.title("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDF files and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done! The documents have been processed and stored.")
            else:
                st.warning("Please upload at least one PDF file.")

    # Main chat interface
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                user_input(prompt)
                # In a real app, you'd get the response here and append it
                # For this example, the user_input function writes the response directly
                # To properly store history, you would refactor user_input to return the response
                # and then append it to session_state.messages

if __name__ == "__main__":
    main()
