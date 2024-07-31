import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq
from pandasai.llm.openai import OpenAI
from pandasai import SmartDataframe

# Load environment variables
load_dotenv()

# Set up Streamlit page
st.set_page_config(page_title="CA Data Analysis Platform", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #F5F5F5;
    }
    .css-1bc7jzt {
        color: #0072C6;
    }
    .sidebar .sidebar-content {
        background-color: #F5F5F5;
        color: #333333;
    }
    .stButton button {
        background-color: #0072C6;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add a heading
st.markdown("<h1 style='text-align: center; color: #0072C6;'>CA Data Analysis Platform</h1>", unsafe_allow_html=True)

# Define functions to load language models
@st.cache_resource
def load_groq_llm():
    return ChatGroq(model_name="Llama-3.1-70b-Versatile", api_key=os.getenv('GROQ_API_KEY'))

@st.cache_resource
def load_openai_llm():
    return OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Sidebar for user inputs
st.sidebar.title("Settings")
file_type = st.sidebar.radio("Select file type", ("CSV", "Excel"))
if file_type == "CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
else:
    uploaded_file = st.sidebar.file_uploader("Upload an Excel file", type=["xls", "xlsx"])
llm_choice = st.sidebar.selectbox("Select Language Model", ("Groq", "OpenAI"))

# Main application content starts here
if uploaded_file is not None:
    if file_type == "CSV":
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    # Quick preview of the data
    st.subheader("Data Preview")
    st.write(data.head())

    # General Information
    st.subheader("General Information")
    st.write(f"Shape of the dataset: {data.shape}")
    st.write(f"Data Types:\n{data.dtypes}")
    st.write(f"Memory Usage: {data.memory_usage(deep=True).sum()} bytes")

    # Load LLMs
    groq_llm = load_groq_llm()
    openai_llm = load_openai_llm()

    # SmartDataframe setup for language model interaction
    df_groq = SmartDataframe(data, config={'llm': groq_llm})
    df_openai = SmartDataframe(data, config={'llm': openai_llm})

    # User query input for natural language analysis
    query = st.text_input("Enter your query about the data:")
    if query:
        try:
            response = ""
            if llm_choice == "Groq":
                response = df_groq.chat(query)
            elif llm_choice == "OpenAI":
                response = df_openai.chat(query)
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
