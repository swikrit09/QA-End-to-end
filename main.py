import streamlit as st

# Set page config
st.set_page_config(page_title="RAGify", page_icon="🤖", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 3em;
            font-weight: bold;
            color: #ff4b4b;
        }
        .subtitle {
            text-align: center;
            font-size: 1.5em;
            color: #bbb;
        }
        .stButton > button {
            width: 100%;
            padding: 10px;
            font-size: 1.2em;
            border-radius: 10px;
            background-color: #ff4b4b;
            color: white;
            border: none;
            transition:all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #fa4000;
            color: #fff;
        }
    </style>
""", unsafe_allow_html=True)

# Display title and subtitle
st.markdown("<div class='title'>✨ RAGify ✨</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Choose your RAG mode</div>", unsafe_allow_html=True)

st.write("""
### Welcome to RAGify! 🚀

Choose how you want to perform Retrieval-Augmented Generation (RAG):
- **Web RAG** 🌍: Perform RAG on web-based content.
- **Document RAG** 📄: Retrieve and generate from uploaded documents.
""")

# Create columns for buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("🌍 Web RAG"):
        st.switch_page("pages/webRag.py")  # Navigate to Web RAG page

with col2:
    if st.button("📄 Document RAG"):
        st.switch_page("pages/docRag.py")  # Navigate to Document RAG page

# Celebration 🎈
st.balloons()

footer="""<style>
a:link , a:visited{

color: blue;
background-color: transparent;
text-decoration: none;
font-weight:bold;
letter-spacing:2px;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}

</style>
<div class="footer">
<p>Developed with ❤ by <a style='display: block; text-align: center;' href="https://linkedin.com/in/swikrit-shukla" target="_blank">Swikrit Shukla</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
