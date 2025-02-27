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
import pandas as pd
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
import re

# Set page configuration
st.set_page_config(
    page_title="Advanced Thumbnail Analyzer",
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

# Function to get date range based on timeframe
def get_date_range(timeframe):
    now = datetime.utcnow()
    
    if timeframe == "24 hours":
        start_date = now - timedelta(days=1)
    elif timeframe == "48 hours":
        start_date = now - timedelta(days=2)
    elif timeframe == "7 days":
        start_date = now - timedelta(days=7)
    elif timeframe == "15 days":
        start_date = now - timedelta(days=15)
    elif timeframe == "1 month":
        start_date = now - timedelta(days=30)
    elif timeframe == "3 months":
        start_date = now - timedelta(days=90)
    elif timeframe == "1 year":
        start_date = now - timedelta(days=365)
    else:  # lifetime
        return None
    
    # Format as RFC 3339 timestamp
    return start_date.strftime('%Y-%m-%dT%H:%M:%SZ')

# Function to check if video is a Short
def is_youtube_short(duration_str):
    # Convert duration string (e.g., 'PT2M10S') to seconds
    minutes = re.search(r'(\d+)M', duration_str)
    seconds = re.search(r'(\d+)S', duration_str)
    
    total_seconds = 0
    if minutes:
        total_seconds += int(minutes.group(1)) * 60
    if seconds:
        total_seconds += int(seconds.group(1))
    
    # YouTube Shorts are typically <= 60 seconds
    return total_seconds <= 60

# Function to search YouTube videos
def search_youtube_videos(youtube_api_key, intro_text, video_type, max_results, timeframe):
    try:
        youtube = build('youtube', 'v3', developerKey=youtube_api_key)
        
        # Get date range based on timeframe
        published_after = get_date_range(timeframe)
        
        # Prepare search parameters
        search_params = {
            'q': intro_text,
            'part': 'snippet',
            'maxResults': min(max_results, 50),  # API limit is 50
            'type': 'video',
            'order': 'viewCount'  # Sort by view count
        }
        
        # Add published date filter if not lifetime
        if published_after:
            search_params['publishedAfter'] = published_after
        
        # Execute search
        search_response = youtube.search().list(**search_params).execute()
        
        # Get video details including statistics and content details
        video_ids = [item['id']['videoId'] for item in search_response['items']]
        
        if not video_ids:
            return []
        
        video_response = youtube.videos().list(
            part='snippet,statistics,contentDetails',
            id=','.join(video_ids)
        ).execute()
        
        videos = []
        for item in video_response['items']:
            # Check if this is a Short
            is_short = is_youtube_short(item['contentDetails']['duration'])
            
            # Filter based on video type
            if video_type == "All" or \
               (video_type == "Regular Videos" and not is_short) or \
               (video_type == "Shorts" and is_short):
                
                # Extract data
                video_data = {
                    'id': item['id'],
                    'title': item['snippet']['title'],
                    'channel': item['snippet']['channelTitle'],
                    'channel_id': item['snippet']['channelId'],
                    'views': int(item['statistics'].get('viewCount', 0)),
                    'likes': int(item['statistics'].get('likeCount', 0)) if 'likeCount' in item['statistics'] else 0,
                    'comments': int(item['statistics'].get('commentCount', 0)) if 'commentCount' in item['statistics'] else 0,
                    'published_at': item['snippet']['publishedAt'],
                    'thumbnail_url': item['snippet']['thumbnails']['high']['url'],
                    'is_short': is_short,
                    'duration': item['contentDetails']['duration']
                }
                videos.append(video_data)
        
        # Calculate outlier scores
        videos = calculate_outlier_scores(youtube, videos)
        
        return videos
        
    except Exception as e:
        st.error(f"Error searching YouTube videos: {e}")
        return []

# Function to calculate outlier scores
def calculate_outlier_scores(youtube, videos):
    try:
        # Group videos by channel
        channels = {}
        for video in videos:
            channel_id = video['channel_id']
            if channel_id not in channels:
                channels[channel_id] = {'regular': [], 'shorts': []}
            
            if video['is_short']:
                channels[channel_id]['shorts'].append(video)
            else:
                channels[channel_id]['regular'].append(video)
        
        # Calculate avg views per channel
        for channel_id, data in channels.items():
            # Get channel data (more videos if needed)
            try:
                channel_response = youtube.channels().list(
                    part='statistics',
                    id=channel_id
                ).execute()
                
                channel_stats = channel_response['items'][0]['statistics']
                total_videos = int(channel_stats.get('videoCount', 0))
                total_views = int(channel_stats.get('viewCount', 0))
                
                # Approximate average views
                avg_views = total_views / total_videos if total_videos > 0 else 0
                
                # Set avg_views for both video types
                channels[channel_id]['avg_views'] = avg_views
            except Exception as e:
                # If can't get channel stats, use average of current videos
                regular_avg = sum(v['views'] for v in data['regular']) / len(data['regular']) if data['regular'] else 0
                shorts_avg = sum(v['views'] for v in data['shorts']) / len(data['shorts']) if data['shorts'] else 0
                
                channels[channel_id]['avg_views'] = regular_avg
        
        # Calculate outlier scores
        for video in videos:
            channel_id = video['channel_id']
            avg_views = channels[channel_id]['avg_views']
            
            if avg_views > 0:
                video['outlier_score'] = video['views'] / avg_views
            else:
                video['outlier_score'] = 1.0
        
        return videos
        
    except Exception as e:
        st.error(f"Error calculating outlier scores: {e}")
        # Return videos without outlier scores
        for video in videos:
            video['outlier_score'] = 1.0
        return videos

# Function to download and analyze thumbnails
def analyze_thumbnails(videos, vision_client, openai_client):
    results = []
    
    for video in videos:
        try:
            # Download thumbnail
            response = requests.get(video['thumbnail_url'])
            img = Image.open(io.BytesIO(response.content))
            
            # Convert to bytes for API processing
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Process with Google Vision API
            vision_results = None
            if vision_client:
                vision_results = analyze_with_vision(img_byte_arr, vision_client)
            
            # Process with OpenAI
            base64_image = encode_image(img_byte_arr)
            openai_description = analyze_with_openai(openai_client, base64_image)
            
            # Generate prompt paragraph
            prompt = None
            if vision_results:
                prompt = generate_prompt_paragraph(openai_client, vision_results, openai_description)
            else:
                prompt = generate_prompt_paragraph(openai_client, {"no_vision_api": True}, openai_description)
            
            # Add results
            results.append({
                'video': video,
                'vision_results': vision_results,
                'openai_description': openai_description,
                'prompt': prompt,
                'thumbnail_image': img
            })
        
        except Exception as e:
            st.error(f"Error analyzing thumbnail for video {video['id']}: {e}")
    
    return results

# Function to generate optimal thumbnail prompt based on multiple analyses
def generate_optimal_prompt(client, thumbnail_analyses, intro_text):
    try:
        # Extract prompts and video stats
        analysis_data = []
        for analysis in thumbnail_analyses:
            analysis_data.append({
                'prompt': analysis['prompt'],
                'views': analysis['video']['views'],
                'outlier_score': analysis['video']['outlier_score'],
                'is_short': analysis['video']['is_short']
            })
        
        prompt = f"""
        You are a YouTube thumbnail expert. I need you to analyze multiple successful thumbnails and generate ONE optimal thumbnail design guideline.
        
        Here's the intro/description of the video I want to create: "{intro_text}"
        
        Below are analyses of {len(analysis_data)} successful YouTube thumbnails in this niche, along with their view counts and outlier scores:
        
        {json.dumps(analysis_data, indent=2)}
        
        Based on these analyses and the video intro I provided, create a SINGLE COHESIVE PARAGRAPH that describes the optimal thumbnail design for my video.
        
        Your paragraph should:
        1. Identify common patterns among the most successful thumbnails (highest views and outlier scores)
        2. Suggest specific colors, layout, text, and visual elements 
        3. Describe how these elements should be arranged
        4. Explain how the thumbnail should capture the essence of my video intro
        5. Include emotional triggers that will maximize click-through rates
        
        Make this paragraph comprehensive and specific enough that a designer could create the thumbnail from your description.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a YouTube thumbnail expert with deep knowledge of what drives high click-through rates."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating optimal prompt: {e}")
        return None

# Main app
def main():
    st.title("Advanced YouTube Thumbnail Analyzer")
    st.write("Analyze thumbnails, find successful videos, and generate optimal thumbnail designs.")
    
    # Initialize and check API clients
    vision_client, openai_client, youtube_api_key = setup_credentials()
    
    if not openai_client:
        st.error("OpenAI client not initialized. Please check your API key.")
        return
    
    if not youtube_api_key:
        st.error("YouTube API key not provided. Please enter a valid API key.")
        return
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Single Thumbnail Analysis", "YouTube Search & Analysis"])
    
    # Tab 1: Single Thumbnail Analysis
    with tab1:
        st.header("Analyze a Single Thumbnail")
        
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
    
    # Tab 2: YouTube Search & Analysis
    with tab2:
        st.header("Find & Analyze Similar Videos")
        
        # Video intro input
        intro_text = st.text_area("Enter your video intro/description:", height=100)
        
        # Search configuration
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            video_type = st.selectbox(
                "Content Type",
                ["All", "Regular Videos", "Shorts"]
            )
        
        with col2:
            timeframe = st.selectbox(
                "Upload Timeframe",
                ["24 hours", "48 hours", "7 days", "15 days", "1 month", "3 months", "1 year", "Lifetime"]
            )
        
        with col3:
            max_results = st.number_input("Number of Results", min_value=1, max_value=10, value=5)
        
        with col4:
            sort_by = st.selectbox(
                "Sort Results By",
                ["Views", "Outlier Score"]
            )
        
        # Search button
        search_button = st.button("Search YouTube")
        
        if search_button and intro_text:
            with st.spinner("Searching YouTube and analyzing thumbnails..."):
                # Search YouTube
                videos = search_youtube_videos(youtube_api_key, intro_text, video_type, max_results, timeframe)
                
                if not videos:
                    st.warning("No videos found matching your criteria. Try a different search.")
                else:
                    # Sort videos
                    if sort_by == "Views":
                        videos.sort(key=lambda x: x['views'], reverse=True)
                    else:  # Outlier Score
                        videos.sort(key=lambda x: x['outlier_score'], reverse=True)
                    
                    # Analyze thumbnails
                    thumbnail_analyses = analyze_thumbnails(videos, vision_client, openai_client)
                    
                    # Display results
                    st.subheader(f"Found {len(videos)} Videos")
                    
                    # Create columns for videos
                    for i, analysis in enumerate(thumbnail_analyses):
                        video = analysis['video']
                        
                        st.markdown(f"### {i+1}. {video['title']}")
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.image(analysis['thumbnail_image'], caption=f"Thumbnail", use_column_width=True)
                            
                            # Video stats
                            st.markdown(f"**Channel:** {video['channel']}")
                            st.markdown(f"**Views:** {video['views']:,}")
                            st.markdown(f"**Outlier Score:** {video['outlier_score']:.2f}x")
                            st.markdown(f"**Published:** {video['published_at'][:10]}")
                            st.markdown(f"**Type:** {'Short' if video['is_short'] else 'Regular Video'}")
                        
                        with col2:
                            # Thumbnail analysis
                            with st.expander("Thumbnail Analysis", expanded=False):
                                st.markdown(analysis['openai_description'])
                            
                            # Thumbnail prompt
                            with st.expander("Thumbnail Prompt", expanded=True):
                                st.markdown(analysis['prompt'])
                                
                                # Link to video
                                st.markdown(f"[Watch Video on YouTube](https://www.youtube.com/watch?v={video['id']})")
                        
                        st.divider()
                    
                    # Generate optimal thumbnail prompt
                    st.subheader("Optimal Thumbnail Design")
                    with st.spinner("Generating optimal thumbnail design..."):
                        optimal_prompt = generate_optimal_prompt(openai_client, thumbnail_analyses, intro_text)
                        
                        st.markdown("### Based on analysis of all thumbnails:")
                        st.text_area("Copy this optimal prompt:", value=optimal_prompt, height=300)
                        
                        # Add a download button for the optimal prompt
                        st.download_button(
                            label="Download Optimal Prompt",
                            data=optimal_prompt,
                            file_name="optimal_thumbnail_prompt.txt",
                            mime="text/plain"
                        )

if __name__ == "__main__":
    main()
