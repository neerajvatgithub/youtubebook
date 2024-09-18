import streamlit as st
import os
import time
import logging
from videoprocessing import main as process_video, extract_video_id, get_video_info, get_transcript, \
    get_youtube_video_duration, get_openai_api_key, get_google_api_key
from dotenv import load_dotenv
import subprocess
import sys

# Load environment variables
load_dotenv()


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


try:
    import PIL
except ImportError:
    install('pillow')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="Video2TextBook", page_icon="📚", layout="centered")

# Custom CSS
st.markdown("""
<style>
    body {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #2c3e50;
    }
    .stApp {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #3498db;
        font-family: 'Arial', sans-serif;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .stTextInput>div>div>input {
        border: 2px solid #3498db;
        border-radius: 5px;
        padding: 0.5rem;
    }
    .stCheckbox {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

st.title("Video2TextBook")

# Initialize session state
if 'youtube_url' not in st.session_state:
    st.session_state.youtube_url = ''
if 'mcq' not in st.session_state:
    st.session_state.mcq = False
if 'flashcards' not in st.session_state:
    st.session_state.flashcards = False
if 'glossary' not in st.session_state:
    st.session_state.glossary = False
if 'mindmap' not in st.session_state:
    st.session_state.mindmap = False


# Function to create download buttons
def create_download_button(file_path, label, file_name):
    with open(file_path, "rb") as file:
        st.download_button(
            label=label,
            data=file,
            file_name=file_name,
            mime="application/pdf"
        )


# UI Components
youtube_url = st.text_input("Enter YouTube URL", value=st.session_state.youtube_url, key="youtube_url")

st.subheader("Additional Content:")
col1, col2 = st.columns(2)
with col1:
    mcq = st.checkbox("MCQ 📝", value=st.session_state.mcq, key="mcq")
    flashcards = st.checkbox("FlashCards 🗂️", value=st.session_state.flashcards, key="flashcards")
with col2:
    glossary = st.checkbox("Glossary 📚", value=st.session_state.glossary, key="glossary")
    mindmap = st.checkbox("Mindmap 🧠", value=st.session_state.mindmap, key="mindmap")


def reset_inputs():
    st.session_state.youtube_url = ''
    st.session_state.mcq = False
    st.session_state.flashcards = False
    st.session_state.glossary = False
    st.session_state.mindmap = False
    if 'textbook_pdf_path' in st.session_state:
        del st.session_state.textbook_pdf_path
    if 'mcq_pdf_path' in st.session_state:
        del st.session_state.mcq_pdf_path


# Add some vertical spacing
st.write("")
st.write("")

# Use columns for button layout
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    generate_button = st.button("Generate Textbook 🚀", key="generate_button")
with col3:
    reset_button = st.button("Reset", on_click=reset_inputs, key="reset_button")

# Add some vertical spacing after buttons
st.write("")
st.write("")

if generate_button:
    if youtube_url:
        try:
            # Check if API keys are available
            openai_api_key = get_openai_api_key()
            google_api_key = get_google_api_key()

            if not openai_api_key or not google_api_key:
                st.error("API keys are missing. Please check your configuration.")
                st.stop()

            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Extracting video information...")
            video_id = extract_video_id(youtube_url)
            if not video_id:
                st.error("Invalid YouTube URL")
                st.stop()

            video_topic, video_description = get_video_info(youtube_url)
            progress_bar.progress(10)

            status_text.text("Fetching video transcript...")
            transcript = get_transcript(video_id)
            if not transcript:
                st.error("Failed to fetch transcript")
                st.stop()
            progress_bar.progress(20)

            status_text.text("Processing video content...")
            duration = get_youtube_video_duration(youtube_url)
            progress_bar.progress(30)

            additional_content = []
            if mcq:
                additional_content.append("MCQ")
            if flashcards:
                additional_content.append("FlashCards")
            if glossary:
                additional_content.append("Glossary")
            if mindmap:
                additional_content.append("Mindmap")

            status_text.text("Generating textbook content...")
            result, pdf_path, additional_content_results = process_video(youtube_url, os.getcwd(), additional_content)
            progress_bar.progress(80)

            if result == "Success" and pdf_path:
                status_text.text("Finalizing output...")
                progress_bar.progress(100)
                time.sleep(1)
                status_text.text("Textbook generated successfully!")

                filename = os.path.basename(pdf_path)
                st.success(f"Textbook '{filename}' generated successfully!")

                # Store the textbook PDF path in session state
                st.session_state.textbook_pdf_path = pdf_path

                if mcq and additional_content_results and "MCQ" in additional_content_results:
                    mcq_result = additional_content_results["MCQ"]
                    if "pdf_path" in mcq_result:
                        mcq_pdf_path = mcq_result["pdf_path"]
                        if os.path.exists(mcq_pdf_path):
                            # Store the MCQ PDF path in session state
                            st.session_state.mcq_pdf_path = mcq_pdf_path
                        else:
                            st.error(f"MCQ PDF file not found at: {mcq_pdf_path}")
                    elif "error" in mcq_result:
                        st.error(f"Error in MCQ generation: {mcq_result['error']}")
                    else:
                        st.error("Unexpected MCQ result format")
                elif mcq:
                    st.info("MCQ generation was requested but no results were returned.")

                # Display other additional content results
                for content_type, content in additional_content_results.items():
                    if content_type != "MCQ":
                        st.subheader(f"{content_type} Content")
                        st.text_area(f"{content_type} Output", content, height=200)
            else:
                st.error(f"An error occurred during processing: {result}")
                logging.error(f"Processing error: {result}")

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            logging.exception("An unexpected error occurred during processing")
    else:
        st.warning("Please enter a YouTube URL.")

# Display download buttons outside the generate_button block
if 'textbook_pdf_path' in st.session_state:
    create_download_button(
        st.session_state.textbook_pdf_path,
        "Download Textbook 📘",
        os.path.basename(st.session_state.textbook_pdf_path)
    )

if 'mcq_pdf_path' in st.session_state:
    create_download_button(
        st.session_state.mcq_pdf_path,
        "Download MCQ PDF 📝",
        os.path.basename(st.session_state.mcq_pdf_path)
    )

st.markdown("---")
st.markdown("Created with ❤️ by Your Name")