import streamlit as st
import os
import io
import json
import requests
import sqlite3
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
import openai
import base64
import time
from datetime import datetime, timedelta
import re
import hashlib

#######################
# SQLite Caching Setup
#######################

def init_db():
    conn = sqlite3.connect("sessions.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS session_cache (
        session_key TEXT PRIMARY KEY,
        optimal_prompts TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    return conn

def create_session_key(user_text, input_type, region, video_type, timeframe, max_results, sort_by):
    key_str = f"{user_text}_{input_type}_{region}_{video_type}_{timeframe}_{max_results}_{sort_by}"
    return hashlib.md5(key_str.encode('utf-8')).hexdigest()

def get_cached_session(conn, session_key):
    cursor = conn.cursor()
    cursor.execute("SELECT optimal_prompts FROM session_cache WHERE session_key = ?", (session_key,))
    row = cursor.fetchone()
    return row[0] if row else None

def cache_session(conn, session_key, optimal_prompts):
    cursor = conn.cursor()
    cursor.execute("REPLACE INTO session_cache (session_key, optimal_prompts) VALUES (?, ?)",
                   (session_key, optimal_prompts))
    conn.commit()

#######################
# Additional Matching Function
#######################

def compute_match_score(video, query_keywords):
    """Compute a simple match score based on how many of the query keywords appear in the video's title and description."""
    keywords = [kw.strip().lower() for kw in query_keywords.split(",") if kw.strip()]
    text = (video['title'] + " " + video['description']).lower()
    # Count each keyword that appears at least once.
    score = sum(1 for kw in keywords if kw in text)
    return score

#######################
# API & Analysis Functions
#######################

def setup_credentials():
    vision_client = None
    openai_client = None
    youtube_api_key = None

    try:
        if 'GOOGLE_CREDENTIALS' in st.secrets:
            credentials_dict = st.secrets["GOOGLE_CREDENTIALS"]
            if isinstance(credentials_dict, str):
                credentials_dict = json.loads(credentials_dict)
            credentials = service_account.Credentials.from_service_account_info(credentials_dict)
            vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        else:
            if os.path.exists("service-account.json"):
                credentials = service_account.Credentials.from_service_account_file("service-account.json")
                vision_client = vision.ImageAnnotatorClient(credentials=credentials)
            else:
                credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
                if credentials_path and os.path.exists(credentials_path):
                    vision_client = vision.ImageAnnotatorClient()
                else:
                    st.error("Google Vision API credentials not found.")
    except Exception as e:
        st.error(f"Error loading Google Vision API credentials: {e}")

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
            openai.api_key = api_key
            openai_client = openai
    except Exception as e:
        st.error(f"Error setting up OpenAI API: {e}")

    try:
        if 'YOUTUBE_API_KEY' in st.secrets:
            youtube_api_key = st.secrets["YOUTUBE_API_KEY"]
        else:
            youtube_api_key = os.environ.get('YOUTUBE_API_KEY')
            if not youtube_api_key:
                youtube_api_key = st.text_input("Enter your YouTube API key:", type="password")
                if not youtube_api_key:
                    st.warning("Please enter a YouTube API key to continue.")
    except Exception as e:
        st.error(f"Error setting up YouTube API: {e}")

    return vision_client, openai_client, youtube_api_key

def analyze_with_vision(image_bytes, vision_client):
    try:
        image = vision.Image(content=image_bytes)
        label_detection = vision_client.label_detection(image=image)
        text_detection = vision_client.text_detection(image=image)
        face_detection = vision_client.face_detection(image=image)
        logo_detection = vision_client.logo_detection(image=image)
        image_properties = vision_client.image_properties(image=image)
        results = {
            "labels": [{"description": label.description, "score": float(label.score)}
                       for label in label_detection.label_annotations],
            "text": [{"description": text.description, "confidence": float(text.confidence) if hasattr(text, 'confidence') else None}
                     for text in text_detection.text_annotations[:1]],
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

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

def analyze_with_openai(client, base64_image):
    try:
        response = client.ChatCompletion.create(
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

def generate_prompt_paragraph(client, vision_results, openai_description):
    try:
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
        response = client.ChatCompletion.create(
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

def extract_keywords(client, user_text, input_type):
    try:
        if input_type == "Title":
            prompt_text = f"""
Extract the 5-7 most important keywords and phrases from this video title.
Focus on terms that would help find related content on YouTube.
Return only the keywords separated by commas, no explanation.

Title: {user_text}
"""
        else:
            prompt_text = f"""
Extract the 5-7 most important keywords and phrases from this video intro/description.
Focus on terms that would help find related content on YouTube.
Return only the keywords separated by commas, no explanation.

Intro: {user_text}
"""
        response = client.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a keyword extraction specialist."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=100
        )
        keywords = response.choices[0].message.content.strip()
        return keywords
    except Exception as e:
        st.error(f"Error extracting keywords: {e}")
        return user_text

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
    else:
        return None
    return start_date.strftime('%Y-%m-%dT%H:%M:%SZ')

def is_youtube_short(duration_str):
    minutes = re.search(r'(\d+)M', duration_str)
    seconds = re.search(r'(\d+)S', duration_str)
    total_seconds = 0
    if minutes:
        total_seconds += int(minutes.group(1)) * 60
    if seconds:
        total_seconds += int(seconds.group(1))
    return total_seconds < 180

# JSON for channel filtering (finance category)
CHANNELS = {
    "finance": {
        "USA": {
            "Graham Stephan": "UCV6KDgJskWaEckne5aPA0aQ",
            "Mark Tilbury": "UCxgAuX3XZROujMmGphN_scA",
            "Andrei Jikh": "UCGy7SkBjcIAgTiwkXEtPnYg",
            "Humphrey Yang": "UCFBpVaKCC0ajGps1vf0AgBg",
            "Brian Jung": "UCQglaVhGOBI0BR5S6IJnQPg",
            "Nischa": "UCQpPo9BNwezg54N9hMFQp6Q",
            "Newmoney": "Newmoney",
            "I will teach you to be rich": "UC7ZddA__ewP3AtDefjl_tWg"
        },
        "India": {
            "Pranjal Kamra": "UCwAdQUuPT6laN-AQR17fe1g",
            "Ankur Warikoo": "UCHYubNqqsWGTN2SF-y8jPmQ",
            "Shashank Udupa": "UCdUEJABvX8XKu3HyDSczqhA",
            "Finance with Sharan": "UCwVEhEzsjLym_u1he4XWFkg",
            "Akshat Srivastava": "UCqW8jxh4tH1Z1sWPbkGWL4g",
            "Labour Law Advisor": "UCVOTBwF0vnSxMRIbfSE_K_g",
            "Udayan Adhye": "UCLQOtbB1COQwjcCEPB2pa8w",
            "Sanjay Kathuria": "UCTMr5SnqHtCM2lMAI31gtFA",
            "Financially free": "UCkGjGT2B7LoDyL2T4pHsUqw",
            "Powerup Money": "UC_eLanNOt5ZiKkZA2Fay8SA",
            "Shankar Nath": "UCtnItzU7q_bA1eoEBjqcVrw",
            "Wint Weath": "UCggPd3Vf9ooG2r4I_ZNWBzA",
            "Invest aaj for Kal": "UCWHCXSKASuSzao_pplQ7SPw",
            "Rahul Jain": "UC2MU9phoTYy5sigZCkrvwiw"
        }
    }
}

def search_youtube_videos(youtube_api_key, user_text, input_type, video_type, max_results, timeframe, openai_client, region):
    try:
        keywords = extract_keywords(openai_client, user_text, input_type)
        st.info(f"Searching YouTube for: {keywords}")
        published_after = get_date_range(timeframe)
        video_ids = []
        search_url = "https://www.googleapis.com/youtube/v3/search"
        if region == "Global":
            search_params = {
                'q': keywords,
                'part': 'snippet',
                'maxResults': min(max_results * 2, 50),
                'type': 'video',
                'order': 'relevance',
                'key': youtube_api_key,
                'relevanceLanguage': 'en'
            }
            if published_after:
                search_params['publishedAfter'] = published_after
            search_response = requests.get(search_url, params=search_params)
            search_data = search_response.json()
            if 'error' in search_data:
                st.error(f"YouTube API error: {search_data['error']['message']}")
                return []
            if 'items' not in search_data or not search_data['items']:
                st.warning("No videos found matching your search criteria.")
                return []
            video_ids = [item['id']['videoId'] for item in search_data['items'] if 'videoId' in item['id']]
        else:
            channels = list(CHANNELS['finance'][region].values())
            for channel_id in channels:
                if not channel_id.startswith("UC"):
                    continue
                search_params = {
                    'q': keywords,
                    'channelId': channel_id,
                    'part': 'snippet',
                    'maxResults': min(max_results, 50),
                    'type': 'video',
                    'order': 'relevance',
                    'key': youtube_api_key,
                    'relevanceLanguage': 'en'
                }
                if published_after:
                    search_params['publishedAfter'] = published_after
                channel_response = requests.get(search_url, params=search_params)
                channel_data = channel_response.json()
                if 'items' in channel_data:
                    for item in channel_data['items']:
                        if 'videoId' in item['id']:
                            video_ids.append(item['id']['videoId'])
        if not video_ids:
            return []
        video_ids = list(set(video_ids))
        video_ids = video_ids[:max_results]
        videos_url = "https://www.googleapis.com/youtube/v3/videos"
        videos_params = {
            'part': 'snippet,statistics,contentDetails',
            'id': ','.join(video_ids),
            'key': youtube_api_key
        }
        videos_response = requests.get(videos_url, params=videos_params)
        videos_data = videos_response.json()
        videos_list = []
        for item in videos_data.get('items', []):
            duration = item['contentDetails']['duration']
            is_short = is_youtube_short(duration)
            if video_type == "All" or (video_type == "Regular Videos" and not is_short) or (video_type == "Shorts" and is_short):
                statistics = item.get('statistics', {})
                view_count = int(statistics.get('viewCount', 0))
                like_count = int(statistics.get('likeCount', 0)) if 'likeCount' in statistics else 0
                comment_count = int(statistics.get('commentCount', 0)) if 'commentCount' in statistics else 0
                video_data = {
                    'id': item['id'],
                    'title': item['snippet']['title'],
                    'description': item['snippet'].get('description', ''),
                    'channel': item['snippet']['channelTitle'],
                    'channel_id': item['snippet']['channelId'],
                    'views': view_count,
                    'likes': like_count,
                    'comments': comment_count,
                    'published_at': item['snippet']['publishedAt'],
                    'thumbnail_url': item['snippet']['thumbnails']['high']['url'],
                    'is_short': is_short,
                    'duration': duration
                }
                # Compute a match score based on title and description overlap with query keywords
                video_data['match_score'] = compute_match_score(video_data, keywords)
                videos_list.append(video_data)
        # Now sort based on chosen criteria combined with match score:
        if sort_by == "Views":
            videos_list.sort(key=lambda v: v['views'] * (v['match_score'] + 1), reverse=True)
        else:
            videos_list.sort(key=lambda v: v['outlier_score'] * (v['match_score'] + 1), reverse=True)
        return videos_list
    except Exception as e:
        st.error(f"Error searching YouTube videos: {e}")
        return []

def calculate_outlier_scores(youtube_api_key, videos):
    try:
        channels = {}
        for video in videos:
            channel_id = video['channel_id']
            if channel_id not in channels:
                channels[channel_id] = {'regular': [], 'shorts': []}
            if video['is_short']:
                channels[channel_id]['shorts'].append(video)
            else:
                channels[channel_id]['regular'].append(video)
        for channel_id, data in channels.items():
            try:
                channel_url = "https://www.googleapis.com/youtube/v3/channels"
                channel_params = {
                    'part': 'statistics',
                    'id': channel_id,
                    'key': youtube_api_key
                }
                channel_response = requests.get(channel_url, params=channel_params)
                channel_data = channel_response.json()
                if 'items' in channel_data and channel_data['items']:
                    channel_stats = channel_data['items'][0]['statistics']
                    total_videos = int(channel_stats.get('videoCount', 0))
                    total_views = int(channel_stats.get('viewCount', 0))
                    if 'shorts' in data and 'regular' in data:
                        if len(data['shorts']) > 0 and len(data['regular']) > 0:
                            avg_views_shorts = (total_views * 0.2) / (total_videos * 0.3) if total_videos > 0 else 0
                            avg_views_regular = (total_views * 0.8) / (total_videos * 0.7) if total_videos > 0 else 0
                            channels[channel_id]['avg_views_shorts'] = avg_views_shorts
                            channels[channel_id]['avg_views_regular'] = avg_views_regular
                        else:
                            avg_views = total_views / total_videos if total_videos > 0 else 0
                            channels[channel_id]['avg_views_shorts'] = avg_views
                            channels[channel_id]['avg_views_regular'] = avg_views
                    else:
                        avg_views = total_views / total_videos if total_videos > 0 else 0
                        channels[channel_id]['avg_views_shorts'] = avg_views
                        channels[channel_id]['avg_views_regular'] = avg_views
                else:
                    shorts_avg = sum(v['views'] for v in data.get('shorts', [])) / len(data['shorts']) if data.get('shorts', []) else 0
                    regular_avg = sum(v['views'] for v in data.get('regular', [])) / len(data.get('regular', [])) if data.get('regular', []) else 0
                    channels[channel_id]['avg_views_shorts'] = shorts_avg if shorts_avg > 0 else 1
                    channels[channel_id]['avg_views_regular'] = regular_avg if regular_avg > 0 else 1
            except Exception as e:
                shorts_avg = sum(v['views'] for v in data.get('shorts', [])) / len(data['shorts']) if data.get('shorts', []) else 0
                regular_avg = sum(v['views'] for v in data.get('regular', [])) / len(data.get('regular', [])) if data.get('regular', []) else 0
                channels[channel_id]['avg_views_shorts'] = shorts_avg if shorts_avg > 0 else 1
                channels[channel_id]['avg_views_regular'] = regular_avg if regular_avg > 0 else 1
        for video in videos:
            channel_id = video['channel_id']
            if video['is_short']:
                avg_views = channels[channel_id]['avg_views_shorts']
            else:
                avg_views = channels[channel_id]['avg_views_regular']
            video['outlier_score'] = video['views'] / avg_views if avg_views > 0 else 1.0
        return videos
    except Exception as e:
        st.error(f"Error calculating outlier scores: {e}")
        for video in videos:
            video['outlier_score'] = 1.0
        return videos

def analyze_thumbnails(videos, vision_client, openai_client):
    results = []
    for video in videos:
        try:
            response = requests.get(video['thumbnail_url'])
            img = Image.open(io.BytesIO(response.content))
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            vision_results = analyze_with_vision(img_byte_arr, vision_client) if vision_client else None
            base64_image = encode_image(img_byte_arr)
            openai_description = analyze_with_openai(openai_client, base64_image)
            if vision_results:
                prompt = generate_prompt_paragraph(openai_client, vision_results, openai_description)
            else:
                prompt = generate_prompt_paragraph(openai_client, {"no_vision_api": True}, openai_description)
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

#######################
# Optimal Prompt Generation with 4 Variations
#######################

def generate_optimal_prompts(client, thumbnail_analyses, user_text):
    try:
        analysis_data = []
        for analysis in thumbnail_analyses:
            analysis_data.append({
                'prompt': analysis['prompt'],
                'views': analysis['video']['views'],
                'outlier_score': analysis['video']['outlier_score'],
                'is_short': analysis['video']['is_short'],
                'title': analysis['video']['title'],
                'description': analysis['video']['description'][:300] if len(analysis['video']['description']) > 300 else analysis['video']['description']
            })
        base_context = f"""
Below are analyses of {len(analysis_data)} successful YouTube thumbnails in this niche, along with their view counts and outlier scores:
{json.dumps(analysis_data, indent=2)}

Based on these analyses and the following video input:
"{user_text}"

Create a highly actionable, SINGLE COHESIVE PARAGRAPH guideline for designing an optimal thumbnail. The guideline must:
- Describe specific colors (with hex codes if possible)
- Detail layout and composition (including typography, spatial arrangement, and balance)
- Explain emotional triggers and branding elements
- Include every element so that a designer can create the thumbnail exactly from your description.
"""
        variants = [
            "Bold and dynamic design with strong typography and vibrant, contrasting colors.",
            "Minimalist and clean design with subtle details, ample white space, and modern fonts.",
            "Modern and edgy design with innovative layout, striking gradients, and a mix of bold and soft elements.",
            "Creative and artistic design that blends traditional elements with contemporary flair, emphasizing balance and visual harmony."
        ]
        prompts = []
        for idx, style in enumerate(variants):
            variant_prompt = f"Variation {idx+1}: {base_context}\nStyle Instruction: {style}\nCreate a thumbnail design guideline which describes everything needed to produce a thumbnail that encapsulates the video input."
            response = client.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a top-tier YouTube thumbnail designer with deep expertise in creating high CTR thumbnails."},
                    {"role": "user", "content": variant_prompt}
                ],
                max_tokens=1000
            )
            prompts.append(response.choices[0].message.content)
        combined_prompts = "\n\n" + ("-"*80 + "\n\n").join(prompts)
        return combined_prompts
    except Exception as e:
        st.error(f"Error generating optimal prompts: {e}")
        return None

#######################
# Main App
#######################

def main():
    st.title("YouTube Thumbnail Analyzer")
    st.write("Find successful videos, analyze their thumbnails, and generate optimal thumbnail designs.")
    
    vision_client, openai_client, youtube_api_key = setup_credentials()
    conn = init_db()
    
    if not openai_client:
        st.error("OpenAI client not initialized. Please check your API key.")
        return

    input_type = st.selectbox("Select Input Type", ["Title", "Intro"])
    user_text = st.text_area(f"Enter your video {input_type.lower()}:", height=100)
    region_filter = st.selectbox("Select Region", ["Global", "USA", "India"])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        video_type = st.selectbox("Content Type", ["All", "Regular Videos", "Shorts"])
    with col2:
        timeframe = st.selectbox("Upload Timeframe", ["24 hours", "48 hours", "7 days", "15 days", "1 month", "3 months", "1 year", "Lifetime"])
    with col3:
        max_results = st.number_input("Number of Results", min_value=1, max_value=10, value=5)
    with col4:
        sort_by = st.selectbox("Sort Results By", ["Views", "Outlier Score"])
    
    session_key = create_session_key(user_text, input_type, region_filter, video_type, timeframe, max_results, sort_by)
    cached = get_cached_session(conn, session_key)
    
    if youtube_api_key:
        search_button = st.button("Search YouTube")
    else:
        st.warning("YouTube API key is required for searching. Please provide a valid API key.")
        search_button = False

    if search_button and user_text:
        if cached:
            st.info("Loaded cached optimal prompts.")
            optimal_prompts = cached
        else:
            with st.spinner("Searching YouTube and analyzing thumbnails..."):
                videos = search_youtube_videos(youtube_api_key, user_text, input_type, video_type, max_results, timeframe, openai_client, region_filter)
                if not videos:
                    st.warning("No videos found matching your criteria. Try a different search.")
                    return
                if sort_by == "Views":
                    videos.sort(key=lambda v: v['views'] * (v['match_score']+1), reverse=True)
                else:
                    videos.sort(key=lambda v: v['outlier_score'] * (v['match_score']+1), reverse=True)
                thumbnail_analyses = analyze_thumbnails(videos, vision_client, openai_client)
                optimal_prompts = generate_optimal_prompts(openai_client, thumbnail_analyses, user_text)
                cache_session(conn, session_key, optimal_prompts)
        results_tab, optimal_tab = st.tabs(["Video Results", "Optimal Thumbnail Design"])
        with results_tab:
            st.subheader(f"Found {len(videos)} Videos")
            for i, analysis in enumerate(thumbnail_analyses):
                video = analysis['video']
                st.markdown(f"### {i+1}. {video['title']}")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(analysis['thumbnail_image'], caption="Thumbnail", use_column_width=True)
                    st.markdown(f"**Channel:** {video['channel']}")
                    st.markdown(f"**Views:** {video['views']:,}")
                    st.markdown(f"**Outlier Score:** {video['outlier_score']:.2f}x")
                    st.markdown(f"**Published:** {video['published_at'][:10]}")
                    st.markdown(f"**Type:** {'Short' if video['is_short'] else 'Regular Video'}")
                with col2:
                    st.markdown("**Thumbnail Analysis:**")
                    st.markdown(analysis['prompt'])
                    st.markdown(f"[Watch Video on YouTube](https://www.youtube.com/watch?v={video['id']})")
                st.divider()
        with optimal_tab:
            st.subheader("Optimal Thumbnail Design Variations")
            with st.spinner("Generating optimal thumbnail design variations..."):
                st.markdown("### Based on analysis of all thumbnails:")
                st.text_area("Copy these optimal prompt variations:", value=optimal_prompts, height=400)
                st.download_button(
                    label="Download Optimal Prompts",
                    data=optimal_prompts,
                    file_name="optimal_thumbnail_prompts.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()
