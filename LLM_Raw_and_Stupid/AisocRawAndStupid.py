# import streamlit as st
# from groq import Groq
# from sentence_transformers import SentenceTransformer
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# import faiss
# import numpy as np
# import os
# import tempfile

# # Function to initialize Groq client with API key
# def initialize_groq(api_key):
#     return Groq(api_key = api_key)

# #function to load PDF and extract text
# def get_groq_response(client, context, question, model_name="llama-3.1-8b-instant"):
#     prompt = f"""  
#     Based on the following context, please answer the question in a concise manner. 
#     Context: {context}
#     Question: {question}
#     Answer: Provide a clear, accurate answer based only on the information in the context.
#     """
 
#     try:
#         response = client.chat.completions.create(
#             model=model_name,
#             messages=[
#                 {"role": "user", "content": prompt}
#                 ],
#             max_tokens=500,
#             temperature=0.0
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e: 
#         return f"Error getting resposne: {str(e)}"
        
# #function to split text into chunks
# def load_embedding_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# #class to store and manage embeddings, create faiss index, so it can speed up similarity search
# class LocalVectorStore:
#     def __init__(self, embedding_model):
#         self.embedding_model = embedding_model 
#         self.chunks = []
#         self.embeddings = None 
#         self.index = None

#     def add_documents(self, documents): 
#         self.chunks = [doc.page_content for doc in documents]

#         embeddings = self.embedding_model.encode(self.chunks)
#         self.embeddings = np.array(embeddings).astype('float32')

#         dimension = self.embeddings.shape[1]
#         self.index = faiss.IndexFlatL2(dimension)
#         self.index.add(self.embeddings)

#     def similarity_search(self, query, k=4):
#         if self.index is None:
#             return []
        
#         query_embedding = self.embedding_model.encode([query])
#         query_embedding = np.array(query_embedding).astype('float32')

#         distances, indices = self.index.search(query_embedding, k)
#         results = []

#         for i in indices[0]:
#             if i < len(self.chunks):
#                 results.append(self.chunks[i])
        
#         return results
    

# def load_and_split_pdf(uploaded_file):
#     # with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
#     #     temp_file.write(uploaded_file.getvalue())
#     #     temp_file_path = temp_file.name

#     # try:
#     #     loader = PyPDFLoader(temp_file_path)
#     #     documents = loader.load()

#     #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     #     split_docs = text_splitter.split_documents(documents)

#     #     os.remove(temp_file_path)  # Clean up the temporary file
#     #     return split_docs 
#     # except Exception as e:
#     #     os.remove(temp_file_path)


#     with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
#         temp_file.write(uploaded_file.getvalue())
#         temp_file_path = temp_file.name

#     try:
#         loader = PyPDFLoader(temp_file_path)
#         documents = loader.load()

#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000, 
#             chunk_overlap=200
#         )
#         split_docs = text_splitter.split_documents(documents)
#         return split_docs 

#     except Exception as e:
#         raise RuntimeError(f"❌ PDF load failed: {e}")

#     finally:
#         os.remove(temp_file_path)

# def process_document(uploaded_file, groq_client, embedding_model):
#     st.write("Processing document...")
 
#     with st.spinner("Reading PDF..."):
#         chunks = load_and_split_pdf(uploaded_file)

#     if not chunks:
#         st.error("Failed to read or split the PDF document.")
#         return
    
#     st.success(f"Loaded {len(chunks)} chunks from the document.")

#     with st.spinner("Creating vector store..."):
#         vector_store = LocalVectorStore(embedding_model)
#         vector_store.add_documents(chunks)

#     st.success("Vector store created successfully.")

#     st.session_state.vector_store = vector_store
#     st.session_state.groq_client = groq_client
#     st.session_state.ready = True


#     if st.session_state.get("ready", False):
#         st.header("Ask Your Questions")

#         st.write("**Try asking:**")
#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("What is this document about?"):
#                 st.session_state.question = "What is this document about?" 
#             if st.button("Who are the main authors or people mentioned?"):
#                 st.session_state.question = "Who are the main authors or people mentioned?"
#         with col2:
#             if st.button("What are the key findings or conclusions?"):
#                 st.session_state.question = "What are the key findings or conclusions?"
#             if st.button("Can you summarize the main points?"):
#                 st.session_state.question = "What are summarize the main points?"


#         question = st.text_input("Enter your question here:", value=st.session_state.get("question", ""))

#         if question:
#             try:
#                 with st.spinner("Searching for relevant context..."):
#                     relevant_chunks = st.session_state.vector_store.similarity_search(question, k=4)
#                     if not relevant_chunks:
#                         st.warning("No relevant context found for your question.")
#                         return
                    
#                     context = "\n\n".join(relevant_chunks)

#                     answer = get_groq_response(
#                         st.session_state.groq_client,
#                         context,
#                         question,
#                         st.session_state.get("model_name", "llama-3.1-8b-instant")  
#                     )

#                 st.write("**Answer:**")
#                 st.write(answer)

#                 st.success("Answer generated successfully.")
#                 with st.expander("📚 View source chunks"):
#                     for i, chunk in enumerate(relevant_chunks):
#                         st.write(f"**Chunk {i+1}:**")
#                         # Truncate long chunks for readability
#                         display_chunk = chunk[:400] + "..." if len(chunk) > 400 else chunk
#                         st.write(display_chunk)
#                         st.write("---")

#             except Exception as e: 
#                 # Handle any errors that occur during processing
#                 if "rate limit" in str(e).lower():
#                     st.error("Rate limit exceeded. Please wait a moment and try again.")
#                     st.info("💡 Free tier limits are generous but not unlimited!")
#                 else:
#                     st.error(f"❌ Error: {str(e)}")
#                     st.info("💡 Try simplifying your question or check your API key")


# def main():
#     st.set_page_config(
#         page_title="Aisoc Raw and Stupid",
#         page_icon=":robot:",
#         layout="wide"
#     )
#     st.title("Aisoc Raw and Stupid")
#     st.write("100% free APIs, Upload a PDF and ask questions about it!")

#     st.sidebar.header("Settings")
#     st.sidebar.write("Configure your API settings and model preferences. Get your free API at groq.com")


#     groq_api_key = st.sidebar.text_input(
#         "Groq API Key",
#         type="password",
#         help="Get your free API key at [groq.com](https://groq.com)"
#     )

#     model_options = {
#         "llama-3.1-8b-instant": "Llama 3.1 8B (Fast & Smart)",
#         "llama-3.3-70b-versatile": "Llama 3.3 70B (Most Capable)",
#         "gemma2-9b-it": "Gemma2 9B (Balanced)"
#     }

#     selected_model = st.sidebar.selectbox(
#         "Select Model",
#         options = list(model_options.keys()),
#         format_func=lambda x: model_options[x],
#         index = 0
#     )

#     if not groq_api_key:
#         st.sidebar.warning("Please enter your Groq API key to proceed.")
#         return
    

#     groq_client = initialize_groq(groq_api_key)
#     embedding_model = load_embedding_model()

#     st.session_state.selected_model = selected_model

#     uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"], help="Upload a PDF file to ask questions about its content")

#     if uploaded_file is not None:
#         process_document(uploaded_file, groq_client, embedding_model)


# if __name__ == "__main__":
#     main()


import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import faiss
import numpy as np
import os, tempfile

# ---------- Caching ----------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def initialize_groq(api_key: str) -> Groq:
    return Groq(api_key=api_key)

# ---------- LLM Call ----------
def get_groq_response(client: Groq, context: str, question: str, model_name: str) -> str:
    messages = [
        {"role": "system", "content": "Answer ONLY from the provided context. If unsure, say 'I don't know'."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer concisely:"}
    ]
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=500,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()

# ---------- Vector Store ----------
class LocalVectorStore:
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self.chunks: list[str] = []
        self.embeddings = None
        self.index = None

    def add_documents(self, documents):
        self.chunks = [doc.page_content for doc in documents]
        emb = self.embedding_model.encode(self.chunks)
        self.embeddings = np.array(emb, dtype='float32')
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def similarity_search(self, query: str, k: int = 4):
        if self.index is None:
            return [], []
        q_emb = self.embedding_model.encode([query])
        q_emb = np.array(q_emb, dtype='float32')
        distances, indices = self.index.search(q_emb, k)
        results = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        dists = [float(d) for d in distances[0][:len(results)]]
        return results, dists

# ---------- PDF Load & Split ----------
def load_and_split_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name
    try:
        docs = PyPDFLoader(path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(docs)
    except Exception as e:
        raise RuntimeError(f"❌ PDF load failed: {e}")
    finally:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

# ---------- Flow per document ----------
def process_document(uploaded_file, groq_client, embedding_model):
    with st.spinner("Reading & splitting PDF..."):
        chunks = load_and_split_pdf(uploaded_file)
    if not chunks:
        st.error("Failed to read or split the PDF document.")
        st.stop()

    st.success(f"Loaded {len(chunks)} chunks.")
    with st.spinner("Indexing embeddings..."):
        store = LocalVectorStore(embedding_model)
        store.add_documents(chunks)

    st.session_state.vector_store = store
    st.session_state.groq_client = groq_client
    st.session_state.ready = True

    st.header("Ask Your Questions")
    st.write("**Try asking:**")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("What is this document about?"):
            st.session_state.question = "What is this document about?"
        if st.button("Who are the main authors or people mentioned?"):
            st.session_state.question = "Who are the main authors or people mentioned?"
    with c2:
        if st.button("What are the key findings or conclusions?"):
            st.session_state.question = "What are the key findings or conclusions?"
        if st.button("Can you summarize the main points?"):
            st.session_state.question = "Can you summarize the main points?"

    q = st.text_input("Enter your question here:", value=st.session_state.get("question", ""))

    if q:
        with st.spinner("Retrieving context..."):
            top_chunks, dists = st.session_state.vector_store.similarity_search(q, k=4)
        if not top_chunks:
            st.warning("No relevant context found.")
            return
        context = "\n\n".join(top_chunks)
        ans = get_groq_response(
            st.session_state.groq_client,
            context,
            q,
            st.session_state.get("selected_model", "llama-3.1-8b-instant")
        )
        st.subheader("Answer")
        st.write(ans)

        with st.expander("📚 View source chunks"):
            for i, (chunk, d) in enumerate(zip(top_chunks, dists), start=1):
                st.markdown(f"**Chunk {i} (distance: {d:.3f})**")
                st.write(chunk[:600] + ("..." if len(chunk) > 600 else ""))
                st.write("---")

# ---------- Main ----------
def main():
    st.set_page_config(page_title="My Raw and Stupid LLM", page_icon=":robot:", layout="wide")
    st.title("My Raw and Stupid LLM")
    st.write("100% free APIs. Upload a PDF and ask questions about it!")

    st.sidebar.header("Settings")
    groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
    model_options = {
        "llama-3.1-8b-instant": "Llama 3.1 8B (Fast & Smart)",
        "llama-3.3-70b-versatile": "Llama 3.3 70B (Most Capable)",
        "gemma2-9b-it": "Gemma2 9B (Balanced)"
    }
    selected_model = st.sidebar.selectbox("Select Model", list(model_options.keys()), format_func=lambda x: model_options[x])
    st.session_state.selected_model = selected_model

    if not groq_api_key:
        st.sidebar.warning("Enter your Groq API key to proceed.")
        st.stop()

    groq_client = initialize_groq(groq_api_key)
    embedding_model = load_embedding_model()

    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    if uploaded_file:
        process_document(uploaded_file, groq_client, embedding_model)

if __name__ == "__main__":
    main()
