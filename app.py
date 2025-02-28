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

def create_session_key(*args):
    key_str = "_".join(str(arg) for arg in args)
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
# Utility Functions
#######################

def extract_video_id(url):
    # Simple regex to extract video id from common YouTube URL formats
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:&|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

#######################
# Matching & Sorting
#######################

def compute_match_score(video, query):
    # Very simple case-insensitive substring match in the title.
    return 1 if query.lower() in video['title'].lower() else 0

#######################
# API & Analysis Functions
#######################

def setup_credentials():
    vision_client = None
    openai_client = None
    youtube_api_key = None

    try:
        if 'GOOGLE_CREDENTIALS' in st.secrets:
            creds = st.secrets["GOOGLE_CREDENTIALS"]
            if isinstance(creds, str):
                creds = json.loads(creds)
            credentials = service_account.Credentials.from_service_account_info(creds)
            vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        elif os.path.exists("service-account.json"):
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
        labels = vision_client.label_detection(image=image).label_annotations
        texts = vision_client.text_detection(image=image).text_annotations
        faces = vision_client.face_detection(image=image).face_annotations
        logos = vision_client.logo_detection(image=image).logo_annotations
        props = vision_client.image_properties(image=image).image_properties_annotation.dominant_colors.colors
        results = {
            "labels": [{"description": lab.description, "score": float(lab.score)} for lab in labels],
            "text": [{"description": txt.description, "confidence": float(txt.confidence) if hasattr(txt, 'confidence') else None} for txt in texts[:1]],
            "faces": [{"joy": face.joy_likelihood.name,
                       "sorrow": face.sorrow_likelihood.name,
                       "anger": face.anger_likelihood.name,
                       "surprise": face.surprise_likelihood.name} for face in faces],
            "logos": [{"description": logo.description} for logo in logos],
            "colors": [{"color": {"red": color.color.red, "green": color.color.green, "blue": color.color.blue},
                        "score": float(color.score),
                        "pixel_fraction": float(color.pixel_fraction)} for color in props[:5]]
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
                {"role": "user", "content": [
                    {"type": "text", "text": "Analyze this YouTube thumbnail. Describe what you see in detail."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error analyzing image with OpenAI: {e}")
        return None

def generate_prompt_paragraph(client, vision_results, openai_description):
    try:
        input_data = {"vision_analysis": vision_results, "openai_description": openai_description}
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
    # For title search, simply use the title itself.
    return user_text.strip()

def get_date_range(timeframe):
    now = datetime.utcnow()
    mapping = {
        "24 hours": now - timedelta(days=1),
        "48 hours": now - timedelta(days=2),
        "7 days": now - timedelta(days=7),
        "15 days": now - timedelta(days=15),
        "1 month": now - timedelta(days=30),
        "3 months": now - timedelta(days=90),
        "1 year": now - timedelta(days=365)
    }
    start_date = mapping.get(timeframe)
    return start_date.strftime('%Y-%m-%dT%H:%M:%SZ') if start_date else None

def is_youtube_short(duration_str):
    minutes = re.search(r'(\d+)M', duration_str)
    seconds = re.search(r'(\d+)S', duration_str)
    total_seconds = 0
    if minutes:
        total_seconds += int(minutes.group(1)) * 60
    if seconds:
        total_seconds += int(seconds.group(1))
    return total_seconds < 180

#######################
# YouTube Search Functions
#######################

# JSON for finance channels
FINANCE_CHANNELS = {
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

def fetch_video_details(youtube_api_key, video_ids, content_type):
    if not video_ids:
        return []
    videos_url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        'part': 'snippet,statistics,contentDetails',
        'id': ','.join(video_ids),
        'key': youtube_api_key
    }
    response = requests.get(videos_url, params=params)
    data = response.json()
    videos = []
    for item in data.get('items', []):
        duration = item['contentDetails']['duration']
        is_short = is_youtube_short(duration)
        # Filter based on content type:
        if content_type == "Regular Videos" and is_short:
            continue
        if content_type == "Shorts" and not is_youtube_short(duration):
            continue
        statistics = item.get('statistics', {})
        video = {
            'id': item['id'],
            'title': item['snippet']['title'],
            'description': item['snippet'].get('description', ''),
            'channel': item['snippet']['channelTitle'],
            'channel_id': item['snippet']['channelId'],
            'views': int(statistics.get('viewCount', 0)),
            'likes': int(statistics.get('likeCount', 0)) if 'likeCount' in statistics else 0,
            'comments': int(statistics.get('commentCount', 0)) if 'commentCount' in statistics else 0,
            'published_at': item['snippet']['publishedAt'],
            'thumbnail_url': item['snippet']['thumbnails']['high']['url'],
            'is_short': is_short
        }
        video['match_score'] = compute_match_score(video, video['title'])
        videos.append(video)
    return videos

def search_youtube_by_title(youtube_api_key, title, niche, finance_type, upload_time, content_type, sort_by, max_results):
    published_after = get_date_range(upload_time)
    base_url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        'q': title,
        'part': 'snippet',
        'maxResults': max_results * 2,
        'type': 'video',
        'order': 'relevance',
        'key': youtube_api_key,
        'relevanceLanguage': 'en'
    }
    if published_after:
        params['publishedAfter'] = published_after
    response = requests.get(base_url, params=params)
    data = response.json()
    video_ids = []
    if 'items' in data:
        for item in data['items']:
            if 'videoId' in item['id']:
                video_ids.append(item['id']['videoId'])
    video_ids = list(set(video_ids))
    videos = fetch_video_details(youtube_api_key, video_ids, content_type)
    if niche == "Finance":
        allowed_channels = set()
        if finance_type == "All":
            allowed_channels = set(FINANCE_CHANNELS["USA"].values()).union(set(FINANCE_CHANNELS["India"].values()))
        else:
            allowed_channels = set(FINANCE_CHANNELS[finance_type].values())
        videos = [v for v in videos if v['channel_id'] in allowed_channels]
    return videos[:max_results]

def search_youtube_by_urls(youtube_api_key, urls, content_type, upload_time):
    video_ids = []
    for url in urls:
        vid = extract_video_id(url)
        if vid:
            video_ids.append(vid)
    video_ids = list(set(video_ids))
    videos = fetch_video_details(youtube_api_key, video_ids, content_type)
    if upload_time != "Lifetime":
        published_after = get_date_range(upload_time)
        videos = [v for v in videos if v['published_at'] >= published_after]
    return videos

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
                'outlier_score': analysis['video'].get('outlier_score', 1),
                'is_short': analysis['video']['is_short'],
                'title': analysis['video']['title'],
                'description': analysis['video']['description'][:300]
            })
        base_context = f"""
Below are analyses of {len(analysis_data)} successful YouTube thumbnails with their view counts and match scores:
{json.dumps(analysis_data, indent=2)}

Based on these analyses and the following video title:
"{user_text}"

Create a highly actionable, SINGLE COHESIVE PARAGRAPH guideline for designing an optimal thumbnail that encapsulates the essence of the title.
The guideline must:
- Describe specific colors (with hex codes if possible)
- Detail layout and composition (including typography, spatial arrangement, and balance)
- Explain emotional triggers and branding elements
- Include every element so that a designer can exactly recreate the thumbnail.
"""
        variants = [
            "Bold and dynamic design with strong typography and vibrant, contrasting colors.",
            "Minimalist and clean design with subtle details, ample white space, and modern fonts.",
            "Modern and edgy design with an innovative layout, striking gradients, and a mix of bold and soft elements.",
            "Creative and artistic design that blends traditional elements with contemporary flair, emphasizing balance and visual harmony."
        ]
        prompts = []
        for idx, style in enumerate(variants):
            variant_prompt = f"Variation {idx+1}: {base_context}\nStyle Instruction: {style}\nCreate a thumbnail design guideline that describes every element needed to produce a thumbnail that encapsulates the video title."
            response = client.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a top-tier YouTube thumbnail designer with expertise in creating high CTR thumbnails."},
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
# Thumbnail Analysis
#######################

def analyze_thumbnails(videos, vision_client, openai_client):
    results = []
    for video in videos:
        try:
            response = requests.get(video['thumbnail_url'])
            img = Image.open(io.BytesIO(response.content))
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
            vision_results = analyze_with_vision(img_bytes, vision_client) if vision_client else None
            base64_image = encode_image(img_bytes)
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
# Main App
#######################

def main():
    st.title("YouTube Thumbnail Analyzer")
    st.write("Find successful videos, analyze their thumbnails, and generate optimal thumbnail design guidelines.")
    
    # Initialize API clients and cache
    vision_client, openai_client, youtube_api_key = setup_credentials()
    conn = init_db()
    if not openai_client:
        st.error("OpenAI client not initialized. Please check your API key.")
        return

    # Select input type: Title or Video URL
    input_type = st.selectbox("Select Input Type", ["Title", "Video URL"])
    
    if input_type == "Title":
        title_input = st.text_input("Enter video title:")
        niche = st.selectbox("Select Niche", ["Global", "Finance"])
        finance_type = None
        if niche == "Finance":
            finance_type = st.selectbox("Select Finance Type", ["All", "India", "USA"])
    else:
        st.markdown("Enter up to 20 Video URLs:")
        urls = []
        if "url_count" not in st.session_state:
            st.session_state.url_count = 1
        url0 = st.text_input("Video URL 1", key="url0")
        if url0:
            urls.append(url0)
        add_url = st.button("Add another URL")
        if add_url and st.session_state.url_count < 20:
            st.session_state.url_count += 1
        for i in range(1, st.session_state.get("url_count", 1)):
            url_val = st.text_input(f"Video URL {i+1}", key=f"url{i}")
            if url_val:
                urls.append(url_val)

    # Common filters
    upload_time = st.selectbox("Upload Time", ["24 hours", "48 hours", "7 days", "15 days", "1 month", "3 months", "1 year", "Lifetime"])
    content_type = st.selectbox("Content Type", ["All", "Regular Videos", "Shorts"])
    sort_by = st.selectbox("Sort Results By", ["Outlier", "Views"])
    if input_type == "Title":
        max_results = st.number_input("Number of Results", min_value=1, max_value=20, value=5)
    else:
        max_results = len(urls)

    # Create a session key for caching
    if input_type == "Title":
        session_key = create_session_key(title_input, input_type, niche, finance_type if finance_type else "Global", upload_time, max_results, sort_by)
    else:
        session_key = create_session_key(",".join(urls), input_type, "NA", "NA", upload_time, max_results, sort_by)
    cached = get_cached_session(conn, session_key)
    
    if youtube_api_key:
        search_button = st.button("Search YouTube")
    else:
        st.warning("YouTube API key is required for searching. Please provide a valid API key.")
        search_button = False

    if search_button:
        if input_type == "Title" and title_input:
            if cached:
                st.info("Loaded cached optimal prompts.")
                optimal_prompts = cached
                videos = []  # We use cached result for optimal prompts
                thumbnail_analyses = [] 
            else:
                with st.spinner("Searching YouTube..."):
                    videos = search_youtube_by_title(youtube_api_key, title_input, niche, finance_type if niche=="Finance" else None, upload_time, content_type, sort_by, max_results)
                if not videos:
                    st.warning("No videos found matching your criteria. Try a different search.")
                    return
                with st.spinner("Analyzing thumbnails..."):
                    thumbnail_analyses = analyze_thumbnails(videos, vision_client, openai_client)
                optimal_prompts = generate_optimal_prompts(openai_client, thumbnail_analyses, title_input)
                cache_session(conn, session_key, optimal_prompts)
        elif input_type == "Video URL" and urls:
            if cached:
                st.info("Loaded cached optimal prompts.")
                optimal_prompts = cached
                videos = fetch_video_details(youtube_api_key, [extract_video_id(url) for url in urls if extract_video_id(url)], content_type)
                thumbnail_analyses = analyze_thumbnails(videos, vision_client, openai_client)
            else:
                with st.spinner("Fetching video details..."):
                    videos = search_youtube_by_urls(youtube_api_key, urls, content_type, upload_time)
                if not videos:
                    st.warning("No videos found from provided URLs. Please check your URLs.")
                    return
                with st.spinner("Analyzing thumbnails..."):
                    thumbnail_analyses = analyze_thumbnails(videos, vision_client, openai_client)
                optimal_prompts = generate_optimal_prompts(openai_client, thumbnail_analyses, videos[0]['title'])
                cache_session(conn, session_key, optimal_prompts)
        else:
            st.warning("Please provide valid input.")
            return

        results_tab, optimal_tab = st.tabs(["Video Results", "Optimal Thumbnail Design"])
        with results_tab:
            if videos:
                st.subheader(f"Found {len(videos)} Videos")
                for i, analysis in enumerate(thumbnail_analyses):
                    video = analysis['video']
                    st.markdown(f"### {i+1}. {video['title']}")
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(analysis['thumbnail_image'], caption="Thumbnail", use_column_width=True)
                        st.markdown(f"**Channel:** {video['channel']}")
                        st.markdown(f"**Views:** {video['views']:,}")
                        st.markdown(f"**Published:** {video['published_at'][:10]}")
                        st.markdown(f"**Type:** {'Short' if video['is_short'] else 'Regular Video'}")
                    with col2:
                        st.markdown("**Thumbnail Analysis:**")
                        st.markdown(analysis['prompt'])
                        st.markdown(f"[Watch Video on YouTube](https://www.youtube.com/watch?v={video['id']})")
                    st.divider()
            else:
                st.write("No videos to display; only optimal prompt variations are available.")

        with optimal_tab:
            st.subheader("Optimal Thumbnail Design Variations")
            with st.spinner("Generating optimal thumbnail design variations..."):
                st.markdown("### Based on analysis:")
                st.text_area("Copy these optimal prompt variations:", value=optimal_prompts, height=400)
                st.download_button(
                    label="Download Optimal Prompts",
                    data=optimal_prompts,
                    file_name="optimal_thumbnail_prompts.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()
