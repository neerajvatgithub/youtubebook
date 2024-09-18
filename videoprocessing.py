from youtube_transcript_api import YouTubeTranscriptApi
import re
import requests
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
import cv2
import markdown
import subprocess
import os
from pytubefix import YouTube
from pytubefix.cli import on_progress
from openai import OpenAI
from dotenv import load_dotenv
import logging
import streamlit as st

# Load environment variables
load_dotenv()

# Function to get OpenAI API key
def get_openai_api_key():
    return st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

# Function to get Google API key
def get_google_api_key():
    return st.secrets["GOOGLE_API_KEY"] if "GOOGLE_API_KEY" in st.secrets else os.getenv("GOOGLE_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=get_openai_api_key())

# Get Google API key
google_api_key = get_google_api_key()

def extract_video_id(youtube_url):
    query = urlparse(youtube_url)
    if query.hostname == 'youtu.be':  # short url
        return query.path[1:]
    elif query.hostname in ['www.youtube.com', 'youtube.com']:
        if query.path == '/watch':  # regular video URL
            return parse_qs(query.query)['v'][0]
        elif query.path[:7] == '/embed/':  # embed URL
            return query.path.split('/')[2]
    return None

def get_video_info(url):
    video_id = extract_video_id(url)
    if not video_id:
        return "Invalid YouTube URL", ""
    api_url = f'https://www.googleapis.com/youtube/v3/videos?part=snippet&id={video_id}&key={get_google_api_key()}'
    response = requests.get(api_url)
    if response.status_code == 200:
        video_info = response.json()
        if video_info['items']:
            title = video_info['items'][0]['snippet']['title']
            description = video_info['items'][0]['snippet']['description']
            return title, description
        else:
            return "Video not found.", "Description not found"
    else:
        return "API request failed.", "API request failed"

def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        print(f"An error occurred while fetching the transcript: {e}")
        return None

def generate_chapters_with_llm(transcript):
    global client
    client = OpenAI(api_key=get_openai_api_key())  # Reinitialize client
    prompt = "Generate upto 10 concise chapter titles for the following video transcript:\n\n"
    prompt += transcript[:4000]
    prompt += "\n\nProvide upto 10 chapter titles in the following format:\n1. Title 1\n2. Title 2\n3. Title 3\n4. Title 4\n5. Title 5"
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates chapter titles for video transcripts."},
                {"role": "user", "content": prompt}
            ]
        )
        chapter_titles = response.choices[0].message.content.strip().split('\n')
        return [title.split('. ', 1)[1] for title in chapter_titles if '. ' in title]
    except Exception as e:
        print(f"An error occurred while generating chapters: {e}")
        return []

def generate_chapters_with_timestamps(chapter_titles, video_duration):
    chapters = []
    time_interval = video_duration // len(chapter_titles)
    current_time = 0
    for title in chapter_titles:
        chapters.append((current_time, title))
        current_time += time_interval
    return chapters

def process_youtube_video(transcript, duration):
    transcript_text = " ".join([entry['text'] for entry in transcript])
    chapter_titles = generate_chapters_with_llm(transcript_text)
    chapters_with_timestamps = generate_chapters_with_timestamps(chapter_titles, duration)
    return [list(item) for item in chapters_with_timestamps]

def get_youtube_video_duration(youtube_url):
    response = requests.get(youtube_url)
    if response.status_code != 200:
        raise Exception("Could not fetch the page. Check the URL.")
    soup = BeautifulSoup(response.content, 'lxml')
    duration_meta = soup.find('meta', itemprop='duration')
    if duration_meta:
        video_duration_iso = duration_meta.get('content')
        duration_regex = re.compile(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?')
        match = duration_regex.match(video_duration_iso)
        if match:
            hours = int(match.group(1)) if match.group(1) else 0
            minutes = int(match.group(2)) if match.group(2) else 0
            seconds = int(match.group(3)) if match.group(3) else 0
            return (hours * 3600) + (minutes * 60) + seconds
    return 600

def generate_content(transcript, topic_list):
    global client
    client = OpenAI(api_key=get_openai_api_key())  # Reinitialize client
    topics = ', '.join([item[1] for item in topic_list])
    full_transcript = ' '.join([item['text'] for item in transcript])
    prompt = "Generate long chapter contents for the following video transcript and chapters:\n\n"
    prompt += "\n\Provide in the following format \n1. Title 1\n2. Title 2\n3. Title 3\n4. Title 4\n5. Title 5."
    prompt += "Please provide in PDF ready format with bold chapter titles and formatted contents. Prefix each chapter with word 'Chapter' with chapter number. Generate a detailed textbook style content with atleast 2000 words."
    prompt += "TRANSCRIPT:" + full_transcript[:4000]
    prompt += "TOPICS:" + topics
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates chapter content for video transcripts."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"An error occurred while generating chapters: {e}")
        return []

def extract_chapters_and_sections(content):
    chapters = {}
    current_chapter = None
    current_content = []
    lines = content.split('\n')
    for line in lines:
        if line.startswith("Chapter"):
            if current_chapter:
                chapters[current_chapter] = "\n".join(current_content).strip()
            current_chapter = line.strip()
            current_content = []
        else:
            if line.strip():
                current_content.append(line)
    if current_chapter:
        chapters[current_chapter] = "\n".join(current_content).strip()
    return chapters

def generate_markdown_with_chatgpt(chapters, image_dict):
    global client
    client = OpenAI(api_key=get_openai_api_key())  # Reinitialize client
    prompt = """Generate a well-formatted Markdown document for a book about Recurrent Neural Networks. 
    Include a table of contents at the beginning, and ensure each chapter starts with a level 1 header (# Chapter Title).
    Use appropriate headers, lists, and formatting. Add a horizontal rule (---) before each chapter to ensure page breaks.
    Format the content as follows:

    # Table of Contents
    [Generated Table of Contents]

    ---

    # Chapter 1: [Title]
    [Chapter 1 content]

    ---

    # Chapter 2: [Title]
    [Chapter 2 content]

    ... and so on for all chapters.

    Here's the content to format:

    """
    image_keys = list(image_dict.keys())
    image_index = 0
    for chapter, content in chapters.items():
        prompt += f"# {chapter}\n"
        prompt += f"{content}\n\n"
        if image_index < len(image_keys):
            image_path = image_dict[image_keys[image_index]]
            image_path = image_path.replace("\\", "/")
            prompt += f"![Image for Chapter {chapter}]({image_path})\n\n"
            image_index += 1
        prompt += "---\n\n"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates well-formatted Markdown documents for technical books."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=12000,
        n=1,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()

def convert_markdown_to_html(markdown_content, output_dict, video_topic):
    html_content = markdown.markdown(markdown_content)
    css = """
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; font-size: 18px; }
            h1 { page-break-before: always; color: #2c3e50; text-align: center; font-size: 32px; margin-top: 40px; }
            h1:first-of-type { page-break-before: avoid; }
            h2, h3 { color: #34495e; font-size: 26px; margin-top: 30px; }
            p { font-size: 18px; margin-bottom: 15px; }
            ul { font-size: 18px; margin-bottom: 15px; }
            a { color: #007bff; text-decoration: none; }
            a:hover { text-decoration: underline; }
            .toc { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 30px; }
            .toc h2 { font-size: 28px; }
            .toc ul { list-style-type: none; padding-left: 20px; }
            .toc li { margin-bottom: 10px; font-size: 18px; }
            .main-title { font-size: 36px; text-align: center; color: #2c3e50; margin-bottom: 40px; }
        </style>
    """
    html_content = re.sub(r'<h1>Table of Contents</h1>.*?(<h1>(?!Table of Contents))', r'\1', html_content, flags=re.DOTALL)
    chapters = re.findall(r'<h1>((?!Table of Contents).*?)</h1>', html_content)
    toc_html = '<div class="toc"><h2>Table of Contents</h2><ul>'
    for i, title in enumerate(chapters):
        anchor = f'chapter-{i + 1}'
        toc_html += f'<li><a href="#{anchor}">{title}</a></li>'
        html_content = html_content.replace(f'<h1>{title}</h1>', f'<h1 id="{anchor}">{title}</h1>', 1)
    toc_html += "</ul></div>"
    html_output = f"""
    <html>
    <head>
    {css}
    </head>
    <body>
    <div class="main-title">{video_topic}</div>
    <div style="page-break-after: always;"></div>
    {toc_html}
    <div style="page-break-after: always;"></div>
    {html_content}
    </body>
    </html>
    """
    html_file = "output.html"
    with open(html_file, "w", encoding="utf-8") as file:
        file.write(html_output)
    print(f"HTML file generated successfully: {html_file}")
    return html_file

def convert_html_to_pdf(html_file, pdf_output_file):
    pdf_output_file = pdf_output_file if pdf_output_file.endswith(".pdf") else pdf_output_file + ".pdf"
    wkhtmltopdf_command = [
        "wkhtmltopdf",
        "--enable-local-file-access",
        html_file,
        pdf_output_file
    ]
    try:
        subprocess.run(wkhtmltopdf_command, check=True)
        print(f"HTML successfully converted to PDF: {pdf_output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during HTML to PDF conversion: {e}")

def convert_markdown_to_pdf(markdown_content, output_file, output_dict, video_topic):
    html_file = convert_markdown_to_html(markdown_content, output_dict, video_topic)
    convert_html_to_pdf(html_file, output_file)
    os.remove(html_file)
    print(f"Temporary HTML file removed: {html_file}")

def clean_filename(video_topic):
    invalid_chars = r'[<>:"/\\|?*]'
    cleaned_topic = re.sub(invalid_chars, '_', video_topic)
    cleaned_topic = cleaned_topic.replace("_ ", "_")
    cleaned_topic = cleaned_topic.replace(" ", "_")
    cleaned_topic = cleaned_topic[:255]
    return f"{cleaned_topic}.pdf"

def get_youtube_thumbnail(video_id):
    return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"

def download_youtube_video(url, output_path):
    try:
        yt = YouTube(url, on_progress_callback=on_progress)
        ys = yt.streams.get_highest_resolution()
        ys.download(output_path=output_path)
        return ys.default_filename
    except Exception as e:
        print(f"An error occurred: {e}")
        return -1

def extract_frames(video_path, times, output_folder):
    video = cv2.VideoCapture(video_path)
    output_dict = {}
    output_list = []
    for time in times:
        video.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
        success, frame = video.read()
        if success:
            frame_filename = f"{output_folder}/frame_at_{int(time)}_seconds.jpg"
            cv2.imwrite(frame_filename, frame)
            print(f"Frame saved at {frame_filename}")
            output_dict[time] = frame_filename
            output_list.append(frame_filename)
        else:
            print(f"Failed to extract frame at {time} seconds")
    video.release()
    return output_dict, output_list

def process_video(youtube_url, output_path):
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return "Invalid YouTube URL", None

    video_topic, video_description = get_video_info(youtube_url)
    transcript = get_transcript(video_id)
    if not transcript:
        return "Failed to fetch transcript", None

    duration = get_youtube_video_duration(youtube_url)
    topic_list = process_youtube_video(transcript, duration)
    timestamp_list = [item[0] for item in topic_list]
    content = generate_content(transcript, topic_list)
    chapters = extract_chapters_and_sections(content)

    video_path = download_youtube_video(youtube_url, output_path)
    if video_path == -1:
        return "Failed to download video", None

    video_path = os.path.join(output_path, video_path)
    output_dict, output_list = extract_frames(video_path, timestamp_list, output_path)

    markdown_content = generate_markdown_with_chatgpt(chapters, output_dict)
    output_file = clean_filename(video_topic)
    pdf_path = os.path.join(output_path, output_file)
    convert_markdown_to_pdf(markdown_content, pdf_path, output_dict, video_topic)

    return "Success", pdf_path

def generate_additional_content(content_type, transcript, topic_list):
    global client
    client = OpenAI(api_key=get_openai_api_key())  # Reinitialize client
    prompt = f"Generate {content_type} based on the following transcript and topics:\n\n"
    prompt += "Transcript: " + " ".join([item['text'] for item in transcript])[:4000]
    prompt += "\n\nTopics: " + ", ".join([item[1] for item in topic_list])

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant that generates {content_type} for educational content."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"An error occurred while generating {content_type}: {e}")
        return f"Failed to generate {content_type}"

def generate_mcq_content(transcript, topic_list):
    global client
    client = OpenAI(api_key=get_openai_api_key())  # Reinitialize client
    prompt = "Generate 10 multiple-choice questions based on the following transcript and topics. Each question should have 4 options (A, B, C, D) with only one correct answer. Provide the correct answer at the end of all questions."
    prompt += "\n\nTranscript: " + " ".join([item['text'] for item in transcript])[:4000]
    prompt += "\n\nTopics: " + ", ".join([item[1] for item in topic_list])

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates multiple-choice questions for educational content."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"An error occurred while generating MCQs: {e}")
        return None

def convert_mcq_to_markdown(mcq_content):
    markdown_content = "# Multiple Choice Questions\n\n"
    questions = mcq_content.split('\n\n')
    for i, question in enumerate(questions[:-1], 1):
        markdown_content += f"## Question {i}\n\n{question}\n\n"
    markdown_content += "# Answers\n\n" + questions[-1]
    return markdown_content

def convert_markdown_to_MCQpdf(markdown_content, output_path, title):
    html_content = markdown.markdown(markdown_content)
    css = """
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; font-size: 18px; }
        h1 { page-break-before: always; color: #2c3e50; text-align: center; font-size: 32px; margin-top: 40px; }
        h2 { color: #34495e; font-size: 24px; margin-top: 30px; }
        p { margin-bottom: 15px; }
        .mcq { margin-bottom: 20px; }
        .mcq-question { font-weight: bold; }
        .mcq-options { margin-left: 20px; }
        .mcq-answers { margin-top: 30px; page-break-before: always; }
    """
    html_output = f"""
    <html>
    <head>
    <style>{css}</style>
    </head>
    <body>
    <h1>{title}</h1>
    <div style="page-break-after: always;"></div>
    {html_content}
    </body>
    </html>
    """
    html_file = os.path.join(output_path, "temp_mcq.html")
    with open(html_file, "w", encoding="utf-8") as file:
        file.write(html_output)

    output_file = os.path.join(output_path, f"MCQ_{clean_filename(title)}")
    convert_html_to_pdf(html_file, output_file)
    os.remove(html_file)
    print(f"MCQ PDF created successfully: {output_file}")
    return output_file

def main(youtube_url, output_path, additional_content=None):
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return "Invalid YouTube URL", None, {}

    video_topic, video_description = get_video_info(youtube_url)
    transcript = get_transcript(video_id)
    if not transcript:
        return "Failed to fetch transcript", None, {}

    duration = get_youtube_video_duration(youtube_url)
    topic_list = process_youtube_video(transcript, duration)
    content = generate_content(transcript, topic_list)
    chapters = extract_chapters_and_sections(content)

    # Process main video content
    result, pdf_path = process_video(youtube_url, output_path)
    if result != "Success":
        return result, None, {}

    additional_content_results = {}
    if additional_content:
        for content_type in additional_content:
            if content_type == "MCQ":
                try:
                    mcq_content = generate_mcq_content(transcript, topic_list)
                    if mcq_content:
                        mcq_markdown = convert_mcq_to_markdown(mcq_content)
                        mcq_pdf_path = convert_markdown_to_MCQpdf(mcq_markdown, output_path, video_topic)
                        if mcq_pdf_path and os.path.exists(mcq_pdf_path):
                            additional_content_results["MCQ"] = {"pdf_path": mcq_pdf_path}
                            logging.info(f"MCQ PDF generated successfully: {mcq_pdf_path}")
                        else:
                            additional_content_results["MCQ"] = {"error": "Failed to generate MCQ PDF or file not found"}
                            logging.error(f"Failed to generate MCQ PDF or file not found. Path: {mcq_pdf_path}")
                    else:
                        additional_content_results["MCQ"] = {"error": "Failed to generate MCQ content"}
                        logging.error("Failed to generate MCQ content")
                except Exception as e:
                    logging.error(f"Error in MCQ generation: {str(e)}")
                    additional_content_results["MCQ"] = {"error": f"Error in MCQ generation: {str(e)}"}
            else:
                # Handle other additional content types
                content = generate_additional_content(content_type, transcript, topic_list)
                additional_content_results[content_type] = content

    return "Success", pdf_path, additional_content_results