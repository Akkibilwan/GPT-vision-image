import streamlit as st
import os
import io
import json
import requests
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
import openai
import base64
import time
from datetime import datetime, timedelta
import re

# Set page configuration
st.set_page_config(
    page_title="Thumbnail Analyzer",
    page_icon="🔍",
    layout="wide"
)

# Function to setup API credentials
def setup_credentials():
    vision_client = None
    openai_client = None
    youtube_api_key = None
    
    # For Google Vision API
    try:
        if 'GOOGLE_CREDENTIALS' in st.secrets:
            # Use the provided secrets
            credentials_dict = st.secrets["GOOGLE_CREDENTIALS"]
            if isinstance(credentials_dict, str):
                credentials_dict = json.loads(credentials_dict)
            
            credentials = service_account.Credentials.from_service_account_info(credentials_dict)
            vision_client = vision.ImageAnnotatorClient(credentials=credentials)
            st.success("Google Vision API credentials loaded successfully.")
        else:
            # Check for local file
            if os.path.exists("service-account.json"):
                credentials = service_account.Credentials.from_service_account_file("service-account.json")
                vision_client = vision.ImageAnnotatorClient(credentials=credentials)
                st.success("Google Vision API credentials loaded from local file.")
            else:
                # Look for credentials in environment variable
                credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
                if credentials_path and os.path.exists(credentials_path):
                    vision_client = vision.ImageAnnotatorClient()
                    st.success("Google Vision API credentials loaded from environment variable.")
                else:
                    st.error("Google Vision API credentials not found.")
    except Exception as e:
        st.error(f"Error loading Google Vision API credentials: {e}")
    
    # For OpenAI API
    try:
        api_key = None
        if 'OPENAI_API_KEY' in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
            st.success("OpenAI API key loaded successfully.")
        else:
            api_key = os.environ.get('OPENAI_API_KEY')
            if api_key:
                st.success("OpenAI API key loaded from environment variable.")
            else:
                api_key = st.text_input("Enter your OpenAI API key:", type="password")
                if not api_key:
                    st.warning("Please enter an OpenAI API key to continue.")
        
        if api_key:
            openai_client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Error setting up OpenAI API: {e}")
    
    # For YouTube API
    try:
        if 'YOUTUBE_API_KEY' in st.secrets:
            youtube_api_key = st.secrets["YOUTUBE_API_KEY"]
            st.success("YouTube API key loaded successfully.")
        else:
            youtube_api_key = os.environ.get('YOUTUBE_API_KEY')
            if youtube_api_key:
                st.success("YouTube API key loaded from environment variable.")
            else:
                youtube_api_key = st.text_input("Enter your YouTube API key:", type="password")
                if not youtube_api_key:
                    st.warning("Please enter a YouTube API key to continue.")
    except Exception as e:
        st.error(f"Error setting up YouTube API: {e}")
    
    return vision_client, openai_client, youtube_api_key

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

# Main app
def main():
    st.title("Thumbnail Analyzer")
    st.write("Analyze a thumbnail and generate a detailed prompt with Google Vision AI and OpenAI.")
    
    # Initialize and check API clients
    vision_client, openai_client, _ = setup_credentials()
    
    if not openai_client:
        st.error("OpenAI client not initialized. Please check your API key.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a thumbnail image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Uploaded Thumbnail", use_column_width=True)
        
        # Convert to bytes for API processing
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        with st.spinner("Analyzing thumbnail..."):
            # Process with Google Vision API
            vision_results = None
            if vision_client:
                vision_results = analyze_with_vision(img_byte_arr, vision_client)
            
            # Process with OpenAI
            base64_image = encode_image(img_byte_arr)
            openai_description = analyze_with_openai(openai_client, base64_image)
            
            # Generate both analysis and prompt separately
            if vision_results:
                # Generate structured analysis
                analysis = generate_analysis(openai_client, vision_results, openai_description)
                
                # Display the Analysis section
                st.subheader("Detailed Analysis")
                st.markdown(analysis)
                
                # Create a collapsible container for Vision API results
                with st.expander("View Raw Vision API Results"):
                    st.json(vision_results)
                    
                # Generate the specific prompt paragraph in a separate call
                st.subheader("Thumbnail Prompt")
                with st.spinner("Generating specific prompt..."):
                    prompt_paragraph = generate_prompt_paragraph(openai_client, vision_results, openai_description)
                    
                    # Display prompt in a text area for easy copying
                    st.text_area("Copy this prompt:", value=prompt_paragraph, height=200)
                    
                    # Add a download button for just the prompt
                    st.download_button(
                        label="Download Prompt",
                        data=prompt_paragraph,
                        file_name="thumbnail_prompt.txt",
                        mime="text/plain"
                    )
            else:
                # Use only OpenAI description if Vision API is not available
                st.warning("Google Vision API results not available. Analysis will be based only on OpenAI's image understanding.")
                
                # Generate structured analysis
                analysis = generate_analysis(openai_client, {"no_vision_api": True}, openai_description)
                
                # Display the Analysis section
                st.subheader("Detailed Analysis")
                st.markdown(analysis)
                
                # Generate the specific prompt paragraph in a separate call
                st.subheader("Thumbnail Prompt")
                with st.spinner("Generating specific prompt..."):
                    prompt_paragraph = generate_prompt_paragraph(openai_client, {"no_vision_api": True}, openai_description)
                    
                    # Display prompt in a text area for easy copying
                    st.text_area("Copy this prompt:", value=prompt_paragraph, height=200)
                    
                    # Add a download button for just the prompt
                    st.download_button(
                        label="Download Prompt",
                        data=prompt_paragraph,
                        file_name="thumbnail_prompt.txt",
                        mime="text/plain"
                    )

if __name__ == "__main__":
    main()
