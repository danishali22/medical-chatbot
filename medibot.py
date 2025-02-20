import streamlit as st
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"
@st.cache_resource

def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["contex", "question"])
    return prompt

def load_llm(huggingface_repo_id, hf_token):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":hf_token, "max_length":"512"}
    )

    return llm

def main():
    st.title("Ask Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    prompt = st.chat_input("Pass you prompt here")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        CUSTOM_PROMPT_TEMPLATE = """
            Use the pieces of information provided in the context to answer user's question.
            If you dont know the answer, just say that you dont know, dont try to make up an answer. 
            Dont provide anything out of the given context

            Context: {context}
            Question: {question}

            Start the answer directly. No small talk please.
            """
        
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.getenv("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, hf_token=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k":3}),
                return_source_documents = True,
                chain_type_kwargs = {"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_documents = response["source_documents"]

            #Format Response
            formatted_sources = "\n\n".join(
                f"**Document {i+1}:**\n"
                f"**Text:** {doc.page_content}\n"
                f"**Metadata:** {doc.metadata}"
                for i, doc in enumerate(source_documents)
            )

            # Show Result
            result_to_show = f"{result}\n\n**Source Documents:**\n\n{formatted_sources}"


            st.session_state.messages.append({"role": "assistant", "content": result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

if __name__ == "__main__":
    main()

# How to cure cancer?
# What are Canker sores and how to treat them? 
# What is Carbohydrate intolerance?
