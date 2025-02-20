import streamlit as st

def main():
    st.title("Ask Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    prompt = st.chat_input("Pass you prompt here")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = "I am your Medibot!"
        st.session_state.messages.append({"role": "assistant", "content": response})

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

if __name__ == "__main__":
    main()