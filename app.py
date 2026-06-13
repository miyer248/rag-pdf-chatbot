import streamlit as st
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import google.generativeai as genai
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()

# Configure Gemini API
genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Initialize Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")

# Create uploads folder if it doesn't exist
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("📄 PDF Chatbot")

st.title("📄 PDF Chatbot")
st.write("Upload a PDF and extract its text.")

# Upload PDF
uploaded_file = st.file_uploader(
    "Choose a PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ PDF uploaded successfully!")

    # Read the PDF
    reader = PdfReader(file_path)

    extracted_text = ""

    # Extract text from every page
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()

        if text:
            extracted_text += f"\n--- Page {page_num + 1} ---\n"
            extracted_text += text

    # Show extracted text
    st.subheader("📖 Extracted Text")
    st.text_area(
        "Content",
        extracted_text,
        height=300
    )

    # -------------------------------
    # Split text into chunks
    # -------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(extracted_text)
    # for i, chunk in enumerate(chunks):
    #     print(f"\n------ Chunk {i+1} ------")
    #     print(chunk[:500])   # first 500 characters

    # st.subheader("📦 Chunks Created")
    # st.write(f"Total chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        with st.expander(f"Chunk {i + 1}"):
            st.write(chunk)

    # 4. Create embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 5. Generate embeddings
    embeddings = embedding_model.embed_documents(chunks)

    st.subheader("🧠 Embeddings")

    st.write(f"Generated embeddings for {len(embeddings)} chunks.")

    if embeddings:
        st.write(f"Dimension of each embedding: {len(embeddings[0])}")

        st.write("First 10 values of the first embedding:")

        st.write(embeddings[0][:10])

    documents = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()

        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page_num + 1,
                        "source": uploaded_file.name
                    }
                )
            )
    
    documents = text_splitter.split_documents(documents)
    
    st.subheader("📄 Documents stored in Chroma")

    for i, doc in enumerate(documents):
        with st.expander(f"Document {i+1}"):
            st.write(doc.page_content[:1000])

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory="chroma_db"
    )

    st.success("✅ Embeddings stored in ChromaDB successfully!")

    st.subheader("💬 Chat History")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.subheader("💬 Ask Questions")

    question = st.text_input(
        "Ask something about the uploaded PDF:"
    )

    if question:

        results = vector_store.similarity_search_with_score(
            question,
            k=3
        )
        st.write("Number of results:", len(results))

        # for i, doc in enumerate(results):
        #     st.write(f"Result {i+1}:")
        #     st.write(doc.page_content)
        #     st.write("----------------------")

        context = ""

        for doc,score in results:
            st.write("Score:", score)
            st.write(doc.page_content[:500])
            context += doc.page_content + "\n\n"

        for doc,score in results:
            st.write(
                f"📄 Source: {doc.metadata.get('source', 'Unknown')} | "
                f"Page: {doc.metadata.get('page', 'N/A')}"
            )

        st.subheader("📚 Context Sent to Gemini")
        st.text_area("Context", context, height=300)

        history = ""

        for message in st.session_state.messages:
            history += f"{message['role']}: {message['content']}\n"


        # 3. Build the prompt
        prompt = f"""
You are a helpful assistant that answers questions based on the uploaded PDF.

You should use BOTH:
1. The previous conversation history.
2. The retrieved PDF context.

If the user asks follow-up questions like "its", "they", "that", or "those",
use the conversation history to understand what they refer to.

If the answer cannot be found in the context, say:
"I couldn't find that information in the uploaded PDF."

Conversation History:
{history}

Retrieved Context:
{context}

Current Question:
{question}

Answer:
"""
        # 4. Send to Gemini
        response = model.generate_content(prompt)

        # Save user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        # Save assistant response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response.text
            }
        )

        # 5. Display answer
        st.subheader("🤖 Answer")
        st.write(response.text)

        # Save user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        # Save assistant response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response.text
            }
        )

    