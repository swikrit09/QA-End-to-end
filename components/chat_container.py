import streamlit as st
import time
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from utils.translation import get_language_names, translate

def create_chat_container(llm, prompt, type="Document"):
    # Initialize session state variables if not already set.
    if "last_prompt" not in st.session_state:
        st.session_state.last_prompt = ""
    if "translated_text" not in st.session_state:
        st.session_state.translated_text = ""
    if "response" not in st.session_state:
        st.session_state.response = ""
    if "translated_response" not in st.session_state:
        st.session_state.translated_response = ""
    if "conversations" not in st.session_state:
        st.session_state.conversations = []

    # User enters a question.
    user_prompt = st.text_input(f"Enter Your Question From {type}", key="user_prompt")

    # When the user changes the prompt, update the prompt translation and clear stored response.
    if user_prompt and user_prompt != st.session_state.last_prompt:
        st.session_state.last_prompt = user_prompt
        st.session_state.translated_text = translate("auto", "English", user_prompt)
        st.session_state.response = ""
        st.session_state.trans = ""
        st.session_state.translated_response = ""
    if user_prompt=="":
        st.session_state.response = ""
        st.session_state.trans = ""
        st.session_state.translated_response = ""
        st.session_state.translated_text = ""
        

    # Display the translated prompt.
    if st.session_state.translated_text:
        st.write("**Translated Prompt:**", st.session_state.translated_text)

    # The Ask button processes the translated prompt and stores the response in session state.
    if st.button("Ask", key="ask_btn"):
        if st.session_state.get("vectors"):
            selected_embedding = st.sidebar.selectbox(
                "Select Embedding",
                list(st.session_state.vectors.keys()),
                key="embedding_select"
            )
            if selected_embedding:
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors[selected_embedding]["vectors"].as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                start = time.process_time()
                result = retrieval_chain.invoke({'input': st.session_state.translated_text})
                response_time = time.process_time() - start

                st.session_state.response = result['answer']
                st.session_state.conversations.append({
                    "question": user_prompt,
                    "answer": st.session_state.response,
                    "response_time": response_time,
                    "translated_res": None
                })
        else:
            st.error("No vector embeddings found. Please create one first!")

    # Display the answer (if available) along with its response time.
    if st.session_state.response:
        st.write("**Response:**", st.session_state.response)
        last_convo = st.session_state.conversations[-1]
        st.write(f"**Response Time:** {last_convo['response_time']:.2f} seconds")

        # Provide a select box to choose a target language for the response translation.
        target_language = st.selectbox(
            "Translate the answer to:",
            get_language_names(),
            key="translate_lang"
        )
        # Translate the response when a target language is selected.
        if target_language and target_language!="Select":
            st.session_state.translated_response = translate("English", target_language, st.session_state.response)
            st.session_state.conversations[-1]["translated_res"] = st.session_state.translated_response
            st.write("**Translated Response:**", st.session_state.translated_response)

    # Display saved conversations.
    st.subheader("Saved Conversations")
    if st.session_state.conversations:
        for idx, convo in enumerate(st.session_state.conversations):
            with st.expander(f"Conversation {idx + 1}"):
                st.write("**Question:**", convo["question"])
                st.write("**Answer:**", convo["answer"])
                if(convo['translated_res']):
                    st.write("**Translated Response:**", convo['translated_res'])
    else:
        st.write("No conversations saved yet.")
