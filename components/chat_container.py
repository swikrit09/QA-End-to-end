import streamlit as st
import time
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from utils.translation import get_language_names, translate

# Main Q&A interface
def create_chat_container(llm,prompt,type="Document"):
    prompt1 = st.text_input(f"Enter Your Question From {type}")
    if prompt1:
        if st.session_state.vectors:
            selected_embedding = st.sidebar.selectbox("Select Embedding", st.session_state.vectors.keys())
            if selected_embedding:
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors[selected_embedding]["vectors"].as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                # Process the query
                start = time.process_time()
                if len(st.session_state.conversations) == 0 or prompt1 != st.session_state.conversations[-1]["question"]:
                    response = retrieval_chain.invoke({'input': prompt1})
                    response_time = time.process_time() - start
                    # With a Streamlit expander
                    with st.expander("Document Similarity Search"):
                        for i, doc in enumerate(response["context"]):
                            st.write(doc.page_content)
                            st.write("--------------------------------")


                    # Save the conversation
                    st.session_state.conversations.append({"question": prompt1, "answer": response['answer'],"response_time":response_time })

                # Retrieve the last response
                last_response = st.session_state.conversations[-1]["answer"]
                t = st.session_state.conversations[-1]['response_time']
                # Display the response
                st.write(f"Response Time: {t:.2f} seconds")
                st.write(last_response)

                selected_language = st.selectbox(
                    "Translate to",
                    (get_language_names()),
                    index=0,  # Default to the first language in the list
                )
                if selected_language and selected_language != "Select":
                    translation = translate("English", selected_language, last_response)
                    with st.expander("Translated Response"):
                        st.write(translation)

        else:
            st.error("No vector embeddings found. Please create one first!")


    # Display saved conversations
    st.subheader("Saved Conversations")
    if st.session_state.conversations:
        for idx, convo in enumerate(st.session_state.conversations):
            with st.expander(f"Conversation {idx + 1}"):
                st.write(f"**Question:** {convo['question']}")
                st.write(f"**Answer:** {convo['answer']}")
    else:
        st.write("No conversations saved yet.")
