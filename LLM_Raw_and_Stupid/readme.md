# 📄 My Raw and Stupid LLM

This project is a **Streamlit web app** that lets you upload a PDF file, ask questions about its content, and get answers powered by **Groq LLMs** and **semantic search**.  

It demonstrates **Retrieval-Augmented Generation (RAG)** — combining vector embeddings with large language models to answer questions grounded in your documents.

---

## 🚀 Features
- Upload a PDF and process it into text chunks.  
- Generate embeddings with **SentenceTransformers (MiniLM model)**.  
- Store embeddings in a **FAISS vector database** for fast similarity search.  
- Query **Groq LLM API** for answers based only on the PDF’s content.  
- Interactive **Streamlit UI** with pre-set and custom question inputs.  
- Transparency: view the source text chunks used for each answer.  

---

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/) → UI framework  
- [Groq](https://groq.com/) → LLM inference API  
- [SentenceTransformers](https://www.sbert.net/) → Embedding model  
- [FAISS](https://faiss.ai/) → Vector search engine  
- [LangChain](https://www.langchain.com/) → Text splitting + PDF loader  
- [pypdf](https://pypi.org/project/pypdf/) → PDF parsing backend  
- [NumPy](https://numpy.org/) → Array manipulations  

---

## 📂 Project Structure
LLM_Raw_and_Stupid/
│── AisocRawAndStupid.py # Main Streamlit app
│── requirements.txt # Dependencies
│── .gitignore # Git ignore rules
│── README.md # Project documentation

## ⚡ Setup & Usage
1. **Clone this repo**  
   ```bash
   git clone https://github.com/Ojosticka/AI-Summer-of-Code.git
   cd aisoc-raw-and-stupid

2. **Create and activate a virtual environment**
   ```bash
    python -m venv venv
    source venv/bin/activate   # Mac/Linux
    venv\Scripts\activate      # Windows

3. **Install dependencies**
   ```bash
    pip install -r requirements.txt

4. **Run the app**
   ```bash
    streamlit run AisocRawAndStupid.py

5. **Enter your API key**

- Get a free API key from Groq.
- Paste it into the Streamlit sidebar under Settings.

6. **Upload a PDF and start asking questions! 🎉**


🔒 **Environment Variables**
If you prefer to keep your API key safe, create a .env file (ignored by Git):
    ```bash
    GROQ_API_KEY=your_api_key_here
    ```
Then load it in Python with:
   ```bash
    import os
    api_key = os.getenv("GROQ_API_KEY")
   ```

📚 Example Questions

- What is this document about?
- Who are the main authors or people mentioned?
- What are the key findings or conclusions?
- Can you summarize the main points?

🤝 Contributions

This is a learning project for experimenting with LLMs, RAG, and Streamlit.
Contributions, issues, and feedback are always welcome.

📜 License
MIT License – feel free to use and adapt.



