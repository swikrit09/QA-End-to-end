import streamlit as st
import time
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from utils.translation import get_language_names, translate

def create_chat_container(llm, prompt, type="Document"):
    # User enters a question
    prompt1 = st.text_input(f"Enter Your Question From {type}")
    
    if prompt1:
        # If the prompt has changed, clear previous translation
        if ("last_prompt" not in st.session_state) or (st.session_state.last_prompt != prompt1):
            st.session_state.last_prompt = prompt1
            st.session_state.translated_text = None

        # If not already translated, automatically perform translation
        if st.session_state.get("translated_text") is None:
            with st.spinner("Translating..."):
                # The translate function automatically detects the input language using "auto"
                translated_text = translate("auto", "English", prompt1)
            st.session_state.translated_text = translated_text
            # st.success("Translation completed!")
        
        # Display the translated text
        st.write("**Translated Text:**", st.session_state.translated_text)
        
        # The Ask button is enabled automatically once translation is done.
        if st.button("Ask", key="ask_btn"):
            if st.session_state.get("vectors"):
                selected_embedding = st.sidebar.selectbox(
                    "Select Embedding", 
                    list(st.session_state.vectors.keys())
                )
                if selected_embedding:
                    document_chain = create_stuff_documents_chain(llm, prompt)
                    retriever = st.session_state.vectors[selected_embedding]["vectors"].as_retriever()
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)

                    # Process the query and measure response time
                    start = time.process_time()
                    # Only process if this question is new or if no previous conversation exists
                    if (not st.session_state.get("conversations")) or \
                       (prompt1 != st.session_state.conversations[-1]["question"]):
                        response = retrieval_chain.invoke({'input': prompt1})
                        response_time = time.process_time() - start

                        # Display document similarity search details in an expander
                        with st.expander("Document Similarity Search"):
                            for i, doc in enumerate(response["context"]):
                                st.write(doc.page_content)
                                st.write("--------------------------------")

                        # Save the conversation in session state
                        if "conversations" not in st.session_state:
                            st.session_state.conversations = []
                        st.session_state.conversations.append({
                            "question": prompt1, 
                            "answer": response['answer'],
                            "response_time": response_time 
                        })

                    # Retrieve and display the last response
                    last_response = st.session_state.conversations[-1]["answer"]
                    t = st.session_state.conversations[-1]['response_time']
                    st.write(f"**Response Time:** {t:.2f} seconds")
                    st.write(last_response)

                    # Provide a translation option for the response
                    selected_language = st.selectbox(
                        "Translate the answer to:",
                        get_language_names(),
                        index=0  # Default to the first language in the list
                    )
                    if selected_language and selected_language != "Select":
                        translation = translate("English", selected_language, last_response)
                        with st.expander("Translated Response"):
                            st.write(translation)
            else:
                st.error("No vector embeddings found. Please create one first!")

        # Display saved conversations in an expander for better organization
        st.subheader("Saved Conversations")
        if st.session_state.get("conversations"):
            for idx, convo in enumerate(st.session_state.conversations):
                with st.expander(f"Conversation {idx + 1}"):
                    st.write(f"**Question:** {convo['question']}")
                    st.write(f"**Answer:** {convo['answer']}")
        else:
            st.write("No conversations saved yet.")

