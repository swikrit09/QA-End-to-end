import streamlit as st
import os
from utils.available_model import available_models 

def settings_modal():
    """Renders a settings modal with two-column layout for better UI."""

    with st.expander("⚙️ Settings", expanded=False):  # Collapsible settings section
        col1, col2 = st.columns(2)  # Create two columns

        # Left Column: Model Selection
        with col1:
            selected_model = st.selectbox(
                "Choose ChatGroq models to use:",
                options=available_models,
            )
            prompt_template = st.text_area(
            "Enter your custom prompt template:",
            value="""Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
""",
        )

        # st.session_state.groq_api_key = os.getenv('GROQ_API_KEY')
        # st.session_state.google_api_key = os.getenv('GOOGLE_API_KEY')

        # Right Column: API Key Inputs
        with col2:
            groq_api_key = st.text_input(
                "Groq API Key",
                value=st.session_state.get("groq_api_key", ""),
                type="password",
            )

            google_api_key = st.text_input(
                "Google API Key",
                value=st.session_state.get("google_api_key", ""),
                type="password",
            )
            if groq_api_key and google_api_key:
                st.success("API keys loaded successfully!")
            else:
                st.error("Please Add the API Key")

        # Full-width Section for Prompt Template
        
        # Save Settings Button (Full width)
        if st.button("Save Settings"):
            st.session_state.selected_model = selected_model
            st.session_state.prompt_template = prompt_template
            st.session_state.groq_api_key = groq_api_key
            st.session_state.google_api_key = google_api_key

            # Set as environment variables
            os.environ["GROQ_API_KEY"] = groq_api_key
            os.environ["GOOGLE_API_KEY"] = google_api_key

            st.success("Settings updated successfully!")
