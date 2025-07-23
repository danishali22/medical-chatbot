# 🧠 Medical Chatbot with RAG

A Python-based medical chatbot that delivers trustworthy, evidence-backed responses using Retrieval-Augmented Generation (RAG). Built with LangChain, Qdrant vector search, and the Mistral-7B-Instruct model, this project ensures safe and context-bounded answers from reliable medical documents.

## 🚀 Overview

This chatbot is designed to assist users in getting accurate and concise medical information. It combines the power of language models with intelligent document retrieval, ensuring all answers are grounded in trusted content. Ideal for healthcare education, patient awareness, or as a prototype for real-world medical assistant systems.

---

## 🧩 Key Features

- 🔎 **Context-Aware Retrieval**  
  Uses **Qdrant** and **SentenceTransformers** to fetch relevant documents from a medical knowledge base.

- 🧠 **LLM-Powered Answering**  
  Utilizes **Mistral-7B-Instruct** to generate concise, professional, and medically responsible responses.

- 🛡️ **Safe Prompting**  
  Ensures that answers are always based on retrieved context, avoiding hallucinations or false claims.

- 🧱 **Modular Pipeline**  
  Built using **LangChain** and modular Python components for flexibility and easy scaling.

- 🌐 **Streamlit Interface**  
  Clean, interactive chat UI using **Streamlit** for seamless deployment and user interaction.

---

## 🛠️ Technologies Used

- Python  
- LangChain  
- Qdrant  
- Mistral-7B / HuggingFace Transformers  
- SentenceTransformers  
- Streamlit  
- dotenv

---

## 📁 Directory Structure

```
danishali22-medical-chatbot/
├── connect_memory_with_llm.py        # Loads vector DB and handles RAG inference
├── create_memory_for_llm.py          # Parses and embeds medical documents
├── medibot.py                        # Streamlit frontend chat app
├── requirements.txt                  # Python dependencies
└── vectorstore/
    └── db_faiss/
        ├── index.faiss               # FAISS vector index
        └── index.pkl                 # Metadata for FAISS index
```

## ⚙️ Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/danishali22/medical-chatbot.git
   cd medical-chatbot
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Create a `.env` file and add your credentials:

   ```env
   HUGGINGFACE_API_KEY=your_key
   QDRANT_URL=http://localhost:6333
   ```

4. **Run the app**

   ```bash
   streamlit run medibot.py
   ```

---

## ⚠️ Limitations

- This is an educational prototype and **not a substitute for professional medical advice**.
- It relies solely on its document corpus and won’t answer anything outside that scope.

---

## 🧠 Learnings

- Built a robust RAG pipeline using LangChain  
- Integrated vector search with Qdrant and SentenceTransformers  
- Applied prompt engineering for medical safety and clarity
