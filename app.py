import streamlit as st
import os
import io
import json
import re
import requests
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
import openai
import base64
import time

# Set page configuration
st.set_page_config(
    page_title="YouTube Thumbnail Analyzer",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for YouTube-like styling with dark mode
st.markdown("""
<style>
    .main {
        background-color: #0f0f0f;
        color: #f1f1f1;
    }
    .stApp {
        background-color: #0f0f0f;
    }
    h1, h2, h3 {
        color: #f1f1f1;
        font-family: 'Roboto', sans-serif;
    }
    p, li, div {
        color: #aaaaaa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #272727;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-weight: 500;
        color: #f1f1f1;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff0000;
        color: white;
    }
    .stButton>button {
        background-color: #ff0000;
        color: white;
        border: none;
        border-radius: 2px;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
        background-color: #121212;
        color: #f1f1f1;
        border: 1px solid #303030;
    }
    .stTextArea>div>div>textarea {
        background-color: #121212;
        color: #f1f1f1;
    }
    .thumbnail-container {
        border: 1px solid #303030;
        border-radius: 8px;
        padding: 10px;
        background-color: #181818;
    }
    .stExpander {
        background-color: #181818;
        border: 1px solid #303030;
    }
    .stAlert {
        background-color: #181818;
        color: #f1f1f1;
    }
    .stMarkdown {
        color: #f1f1f1;
    }
    /* Fix for radio buttons and other controls */
    .stRadio label {
        color: #f1f1f1 !important;
    }
    .stSpinner > div {
        border-top-color: #f1f1f1 !important;
    }
    /* Code blocks and JSON display */
    pre {
        background-color: #121212 !important;
    }
    code {
        color: #a9dc76 !important;
    }
</style>
""", unsafe_allow_html=True)

# Function to setup API credentials
def setup_credentials():
    vision_client = None
    openai_client = None
    
    # For Google Vision API
    try:
        if 'GOOGLE_CREDENTIALS' in st.secrets:
            # Use the provided secrets
            credentials_dict = st.secrets["GOOGLE_CREDENTIALS"]
            if isinstance(credentials_dict, str):
                credentials_dict = json.loads(credentials_dict)
            
            credentials = service_account.Credentials.from_service_account_info(credentials_dict)
            vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        else:
            # Check for local file
            if os.path.exists("service-account.json"):
                credentials = service_account.Credentials.from_service_account_file("service-account.json")
                vision_client = vision.ImageAnnotatorClient(credentials=credentials)
            else:
                # Look for credentials in environment variable
                credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
                if credentials_path and os.path.exists(credentials_path):
                    vision_client = vision.ImageAnnotatorClient()
                else:
                    st.info("Google Vision API credentials not found. Analysis will use only OpenAI.")
    except Exception as e:
        st.info(f"Google Vision API not available: {e}")
    
    # For OpenAI API
    try:
        api_key = None
        if 'OPENAI_API_KEY' in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
        else:
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                api_key = st.text_input("Enter your OpenAI API key:", type="password")
                if not api_key:
                    st.warning("Please enter an OpenAI API key to continue.")
        
        if api_key:
            openai_client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Error setting up OpenAI API: {e}")
    
    return vision_client, openai_client

# Function to get YouTube video ID from URL
def extract_video_id(url):
    # Regular expressions to match different YouTube URL formats
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
    
    youtube_match = re.match(youtube_regex, url)
    if youtube_match:
        return youtube_match.group(6)
    return None

# Function to get thumbnail URL from video ID
def get_thumbnail_url(video_id):
    # Try to get the maxres thumbnail first, then fall back to high quality if not available
    thumbnail_urls = [
        f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/default.jpg"
    ]
    
    for url in thumbnail_urls:
        response = requests.head(url)
        if response.status_code == 200 and int(response.headers.get('Content-Length', 0)) > 1000:
            return url
    
    return None

# Function to download thumbnail from URL
def download_thumbnail(url):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            return response.content
        else:
            return None
    except Exception as e:
        st.error(f"Error downloading thumbnail: {e}")
        return None

# Function to analyze image with Google Vision API
def analyze_with_vision(image_bytes, vision_client):
    try:
        image = vision.Image(content=image_bytes)
        
        # Perform different types of detection
        label_detection = vision_client.label_detection(image=image)
        text_detection = vision_client.text_detection(image=image)
        face_detection = vision_client.face_detection(image=image)
        logo_detection = vision_client.logo_detection(image=image)
        image_properties = vision_client.image_properties(image=image)
        
        # Extract results
        results = {
            "labels": [{"description": label.description, "score": float(label.score)} 
                      for label in label_detection.label_annotations],
            "text": [{"description": text.description, "confidence": float(text.confidence) if hasattr(text, 'confidence') else None}
                    for text in text_detection.text_annotations[:1]],  # Just get the full text
            "faces": [{"joy": face.joy_likelihood.name, 
                       "sorrow": face.sorrow_likelihood.name,
                       "anger": face.anger_likelihood.name,
                       "surprise": face.surprise_likelihood.name}
                     for face in face_detection.face_annotations],
            "logos": [{"description": logo.description} for logo in logo_detection.logo_annotations],
            "colors": [{"color": {"red": color.color.red, 
                                  "green": color.color.green, 
                                  "blue": color.color.blue},
                        "score": float(color.score),
                        "pixel_fraction": float(color.pixel_fraction)}
                      for color in image_properties.image_properties_annotation.dominant_colors.colors[:5]]
        }
        
        return results
    
    except Exception as e:
        st.error(f"Error analyzing image with Google Vision API: {e}")
        return None

# Function to encode image to base64 for OpenAI
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# Function to analyze image with OpenAI
def analyze_with_openai(client, base64_image):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this YouTube thumbnail. Describe what you see in detail."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error analyzing image with OpenAI: {e}")
        return None

# Function to analyze image (structured analysis)
def generate_analysis(client, vision_results, openai_description):
    try:
        # Prepare input for GPT
        input_data = {
            "vision_analysis": vision_results,
            "openai_description": openai_description
        }
        
        prompt = """
        Based on the provided thumbnail analyses from Google Vision AI and your own image reading, create a structured analysis covering:
        - What's happening in the thumbnail
        - Category of video (e.g., gaming, tutorial, vlog) 
        - Theme and mood
        - Colors used and their significance
        - Elements and objects present
        - Subject impressions (emotions, expressions)
        - Text present and its purpose
        - Target audience
        
        Format your response with clear headings and bullet points for easy readability.
        
        Analysis data:
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a thumbnail analysis expert who can create detailed analyses based on image analysis data."},
                {"role": "user", "content": prompt + json.dumps(input_data, indent=2)}
            ],
            max_tokens=800
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating analysis: {e}")
        return None

# Function to generate a specific prompt paragraph
def generate_prompt_paragraph(client, vision_results, openai_description):
    try:
        # Prepare input for GPT
        input_data = {
            "vision_analysis": vision_results,
            "openai_description": openai_description
        }
        
        prompt = """
        Based on the provided thumbnail analyses from Google Vision AI and your own image reading, create a SINGLE COHESIVE PARAGRAPH that very specifically defines the thumbnail.
        
        This paragraph must describe in detail:
        - The exact theme and purpose of the thumbnail
        - Specific colors used and how they interact with each other
        - All visual elements and their precise arrangement in the composition
        - Overall style and artistic approach used in the design
        - Any text elements and exactly how they are presented
        - The emotional impact the thumbnail is designed to create on viewers
        
        Make this paragraph comprehensive and detailed enough that someone could recreate the thumbnail exactly from your description alone.
        DO NOT use bullet points or separate sections - this must be a flowing, cohesive paragraph.
        
        Analysis data:
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a thumbnail description expert who creates detailed, specific paragraph descriptions."},
                {"role": "user", "content": prompt + json.dumps(input_data, indent=2)}
            ],
            max_tokens=800
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating prompt paragraph: {e}")
        return None

# Function to generate prompt variations
def generate_prompt_variations(client, original_prompt):
    try:
        variation_prompt = f"""
        Below is a detailed description of a YouTube thumbnail:
        
        {original_prompt}
        
        Create TWO alternative prompt variations that could describe a thumbnail with the same core message and purpose, but with different visual elements, styles, or approaches.
        
        Each variation should:
        1. Maintain the same overall message and purpose of the thumbnail
        2. Change visual elements, colors, composition, or style
        3. Be a single cohesive paragraph (not bullet points)
        4. Be detailed enough that someone could create the thumbnail from the description
        5. Be optimized for AI image generation (hyper-realistic style)
        
        Label them clearly as "VARIATION 1:" and "VARIATION 2:" and make them distinctly different from each other and from the original.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a creative thumbnail designer who creates varied but purposeful alternatives."},
                {"role": "user", "content": variation_prompt}
            ],
            max_tokens=1200
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating prompt variations: {e}")
        return None

# Function to generate image based on prompt
def generate_image_from_prompt(client, prompt, style="hyper-realistic"):
    try:
        # Enhance the prompt for better image generation
        enhanced_prompt = f"""Create a hyper-realistic YouTube thumbnail based on this description: 
        
        {prompt}
        
        The image should be high-quality, attention-grabbing, and have the polished look of a professional YouTube thumbnail. Make it hyper-realistic with vivid colors and sharp details.
        """
        
        response = client.images.generate(
            model="dall-e-3",
            prompt=enhanced_prompt,
            size="1792x1024",
            quality="hd",
            n=1
        )
        
        # Get the URL of the generated image
        image_url = response.data[0].url
        
        # Download the image
        image_response = requests.get(image_url)
        if image_response.status_code == 200:
            return image_response.content
        else:
            st.error(f"Failed to download generated image: {image_response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

# Main app
def main():
    # Custom header with YouTube-like design (dark mode)
    st.markdown('<div style="display: flex; align-items: center; padding: 10px 0;"><span style="color: #FF0000; font-size: 28px; font-weight: bold; margin-right: 5px;">▶️</span> <h1 style="margin: 0; color: #f1f1f1;">YouTube Thumbnail Analyzer</h1></div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #aaaaaa; margin-top: 0;">Analyze thumbnails using AI to understand what makes them effective</p>', unsafe_allow_html=True)
    
    # Initialize and check API clients
    vision_client, openai_client = setup_credentials()
    
    if not openai_client:
        st.error("OpenAI client not initialized. Please check your API key.")
        return
    
    # Input options
    input_option = st.radio(
        "Select input method:",
        ["Upload Image", "YouTube URL"],
        horizontal=True
    )
    
    image_bytes = None
    image = None
    video_info = {}
    
    if input_option == "Upload Image":
        # File uploader
        uploaded_file = st.file_uploader("Choose a thumbnail image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            
            # Convert to bytes for API processing
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
            image_bytes = img_byte_arr.getvalue()
    
    else:  # YouTube URL
        # YouTube URL input
        youtube_url = st.text_input("Enter YouTube video URL:", placeholder="https://www.youtube.com/watch?v=...")
        
        if youtube_url:
            video_id = extract_video_id(youtube_url)
            if video_id:
                video_info["id"] = video_id
                video_info["url"] = youtube_url
                
                # Get thumbnail URL
                thumbnail_url = get_thumbnail_url(video_id)
                if thumbnail_url:
                    video_info["thumbnail_url"] = thumbnail_url
                    
                    # Download thumbnail
                    image_bytes = download_thumbnail(thumbnail_url)
                    if image_bytes:
                        # Display the thumbnail
                        image = Image.open(io.BytesIO(image_bytes))
                        video_info["title"] = f"Thumbnail for Video ID: {video_id}"
                else:
                    st.error("Could not retrieve thumbnail for this video.")
            else:
                st.error("Invalid YouTube URL. Please enter a valid YouTube video URL.")
    
    # If we have image bytes, process the image
    if image_bytes and image:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="thumbnail-container">', unsafe_allow_html=True)
            st.image(image, caption="Thumbnail" if input_option == "Upload Image" else video_info.get("title", "YouTube Thumbnail"), use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if input_option == "YouTube URL" and "id" in video_info:
                st.markdown(f'<a href="{video_info["url"]}" target="_blank" style="color: #3ea6ff; text-decoration: none; font-weight: 500;">View Original Video</a>', unsafe_allow_html=True)
        
        # Process the image
        with st.spinner("Analyzing thumbnail..."):
            # Process with OpenAI Vision
            base64_image = encode_image(image_bytes)
            openai_description = analyze_with_openai(openai_client, base64_image)
            
            # Process with Google Vision API if available
            vision_results = None
            if vision_client:
                vision_results = analyze_with_vision(image_bytes, vision_client)
            
            # Display analysis
            with col2:
                st.subheader("Thumbnail Analysis")
                
                # Generate structured analysis
                analysis = generate_analysis(openai_client, vision_results if vision_results else {"no_vision_api": True}, openai_description)
                st.markdown(analysis)
                
                if vision_results:
                    with st.expander("View Raw Vision API Results"):
                        st.json(vision_results)
                
                with st.expander("View Raw OpenAI Description"):
                    st.write(openai_description)
            
            # Generate prompt variations
            st.subheader("Thumbnail Prompts")
            tabs = st.tabs(["Original", "Variation 1", "Variation 2"])
            
            with st.spinner("Generating prompts..."):
                # Generate original prompt
                prompt_paragraph = generate_prompt_paragraph(
                    openai_client, 
                    vision_results if vision_results else {"no_vision_api": True}, 
                    openai_description
                )
                
                with tabs[0]:
                    st.subheader("Original Prompt")
                    st.text_area("Copy this prompt:", value=prompt_paragraph, height=150, key="original_prompt")
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.download_button(
                            label="Download Prompt",
                            data=prompt_paragraph,
                            file_name="thumbnail_original_prompt.txt",
                            mime="text/plain"
                        )
                    
                    # Generate image button for original too
                    with col2:
                        if st.button("Generate Image for Original", key="gen_img_0"):
                            with st.spinner("Generating image..."):
                                image_bytes = generate_image_from_prompt(openai_client, prompt_paragraph)
                                if image_bytes:
                                    st.success("Image generated successfully!")
                                    st.image(image_bytes, caption="Generated Thumbnail for Original Prompt", use_column_width=True)
                                    
                                    # Add download button for the image
                                    buffered = io.BytesIO(image_bytes)
                                    st.download_button(
                                        label="Download Image",
                                        data=buffered.getvalue(),
                                        file_name="original_thumbnail.png",
                                        mime="image/png"
                                    )
                
                # Generate variations
                variations = generate_prompt_variations(openai_client, prompt_paragraph)
                
                # Parse variations
                try:
                    variation_parts = variations.split("VARIATION")
                    
                    if len(variation_parts) >= 3:
                        variation1 = variation_parts[1].replace("1:", "").strip()
                        variation2 = variation_parts[2].replace("2:", "").strip()
                        
                        with tabs[1]:
                            st.subheader("Variation 1")
                            st.text_area("Copy this prompt:", value=variation1, height=150, key="variation1")
                            
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                st.download_button(
                                    label="Download Prompt",
                                    data=variation1,
                                    file_name="thumbnail_variation1.txt",
                                    mime="text/plain"
                                )
                            
                            # Generate image button
                            with col2:
                                if st.button("Generate Image for Variation 1", key="gen_img_1"):
                                    with st.spinner("Generating image..."):
                                        image_bytes = generate_image_from_prompt(openai_client, variation1)
                                        if image_bytes:
                                            st.success("Image generated successfully!")
                                            st.image(image_bytes, caption="Generated Thumbnail for Variation 1", use_column_width=True)
                                            
                                            # Add download button for the image
                                            buffered = io.BytesIO(image_bytes)
                                            st.download_button(
                                                label="Download Image",
                                                data=buffered.getvalue(),
                                                file_name="variation1_thumbnail.png",
                                                mime="image/png"
                                            )
                        
                        with tabs[2]:
                            st.subheader("Variation 2")
                            st.text_area("Copy this prompt:", value=variation2, height=150, key="variation2")
                            
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                st.download_button(
                                    label="Download Prompt",
                                    data=variation2,
                                    file_name="thumbnail_variation2.txt",
                                    mime="text/plain"
                                )
                            
                            # Generate image button
                            with col2:
                                if st.button("Generate Image for Variation 2", key="gen_img_2"):
                                    with st.spinner("Generating image..."):
                                        image_bytes = generate_image_from_prompt(openai_client, variation2)
                                        if image_bytes:
                                            st.success("Image generated successfully!")
                                            st.image(image_bytes, caption="Generated Thumbnail for Variation 2", use_column_width=True)
                                            
                                            # Add download button for the image
                                            buffered = io.BytesIO(image_bytes)
                                            st.download_button(
                                                label="Download Image",
                                                data=buffered.getvalue(),
                                                file_name="variation2_thumbnail.png",
                                                mime="image/png"
                                            )
                    else:
                        with tabs[1]:
                            st.warning("Unable to properly parse variation 1.")
                        
                        with tabs[2]:
                            st.warning("Unable to properly parse variation 2.")
                except Exception as e:
                    st.error(f"Error parsing variations: {e}")

if __name__ == "__main__":
    main()
