import streamlit as st
import base64

def display_pdf(file):
    # Opening file from file path

    st.sidebar.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    file.seek(0)
    

    # Embedding PDF in HTML
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="500" type="application/pdf"
                        style="height:400px; width:100%"
                    >
                    </iframe>"""

    # Displaying File
    st.sidebar.markdown(pdf_display, unsafe_allow_html=True)