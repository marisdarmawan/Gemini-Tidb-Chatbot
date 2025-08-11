# Gemini-Powered RAG Chatbot with Streamlit and TiDB Cloud

An interactive chatbot that answers questions based on your uploaded documents. This project leverages the power of Google's Gemini Pro, a Streamlit-based UI, and TiDB Cloud for persistent vector storage, creating a seamless "Chat with your Docs" experience.



---

## ## Core Features 🚀

* **Interactive Chat Interface:** A user-friendly web interface built with Streamlit.
* **PDF Document Upload:** Easily upload one or more PDF documents.
* **Persistent Document Memory:** Documents are processed and their vector embeddings are stored in a TiDB Cloud Serverless cluster. You don't need to re-upload documents every time you use the app.
* **Retrieval-Augmented Generation (RAG):** The chatbot uses information retrieved *directly from your documents* to formulate answers, ensuring responses are accurate and context-aware.
* **Powered by Google Gemini:** Utilizes the `gemini-pro` model for powerful and coherent language understanding and generation.

---

## ## Tech Stack & Architecture 🛠️

This project uses a modern RAG (Retrieval-Augmented Generation) architecture.



* **Frontend:** [Streamlit](https://streamlit.io/)
* **LLM:** [Google Gemini Pro](https://deepmind.google/technologies/gemini/)
* **Vector Database:** [TiDB Cloud](https://tidb.cloud/) (using the `tidb-vector` library)
* **Language:** Python
* **Core Libraries:** [LangChain](https://www.langchain.com/), `google-generativeai`, `PyPDF2`, `SQLAlchemy`

### ### How It Works

1.  **Upload:** The user uploads a PDF file through the Streamlit UI.
2.  **Process & Embed:** The application extracts text from the PDF, splits it into manageable chunks, and uses Google's embedding model to convert each chunk into a numerical vector.
3.  **Store:** These text chunks and their corresponding vectors are stored in a table within a TiDB Cloud database.
4.  **Query & Retrieve:** When a user asks a question, the app embeds the question and queries TiDB to find the most semantically similar text chunks from the stored documents.
5.  **Generate:** The retrieved chunks (the context) and the user's original question are passed to the Gemini Pro model, which generates a final, context-based answer.

---

## ## Getting Started ⚙️

Follow these instructions to set up and run the project locally.

### ### Prerequisites

* Python 3.9+
* A [TiDB Cloud](https://tidbcloud.com/) account with a Serverless cluster.
* A [Google API Key](https://aistudio.google.com/) with the Gemini API enabled.

### ### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure your secrets:**
    * Create a directory `.streamlit` in the root of the project.
    * Inside `.streamlit`, create a file named `secrets.toml`.
    * Add your credentials to this file:
        ```toml
        # .streamlit/secrets.toml

        GOOGLE_API_KEY = "your-google-api-key-here"
        TIDB_CONNECTION_STRING = "mysql+pymysql://user:password@host/dbname"
        ```
    * **Important:** The `secrets.toml` file is included in `.gitignore` to ensure your keys are not committed to the repository.

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

---

## ## Future Improvements 📝

* [ ] Support for more document types (e.g., `.txt`, `.docx`).
* [ ] Add conversation history for follow-up questions.
* [ ] Implement user authentication to manage documents on a per-user basis.
* [ ] Optimize the retrieval process for very large document sets.

---

## ## License 📄

This project is licensed under the MIT License. See the `LICENSE` file for details.
