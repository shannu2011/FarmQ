import os
import base64
import re
import pandas as pd
from typing import List

import streamlit as st
from deep_translator import GoogleTranslator
from serpapi import GoogleSearch
import boto3
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
load_dotenv()

# ---------------------------
# Environment Variables
# ---------------------------
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

# Initialize Polly
polly_client = None
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    polly_client = boto3.client(
        "polly",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

# ---------------------------
# Sidebar Settings
# ---------------------------
st.sidebar.header("Settings")
lang_options = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Marathi": "mr",
    "Kannada": "kn",
    "Gujarati": "gu",
    "Bengali": "bn"
}
selected_lang_name = st.sidebar.selectbox("Input Language", list(lang_options.keys()))
selected_lang_code = lang_options[selected_lang_name]

# ---------------------------
# Polly Voice Mapping
# ---------------------------
polly_voices = {
    "en": "Aditi", "hi": "Aditi", "ta": "Aditi", "te": "Aditi",
    "mr": "Aditi", "kn": "Aditi", "gu": "Aditi", "bn": "Aditi"
}

# ---------------------------
# Load Embedding Model
# ---------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_model()

# ---------------------------
# Load CSV for extra domain enrichment
# ---------------------------
try:
    df_csv = pd.read_csv("Agri_dataset.csv")
    st.sidebar.success("CSV loaded for domain enrichment")
except:
    df_csv = pd.DataFrame()
    st.sidebar.warning("CSV not loaded, continuing without it")

# ---------------------------
# Define base domains
# ---------------------------
AGRI_DOMAINS = {
    "Soil": "soil quality, soil fertility, moisture, nutrients",
    "Seed Quality": "seed germination, seed quality, seed planting",
    "Irrigation": "water supply, irrigation, drought, flooding",
    "Pests": "pest attacks, insects, locusts, pest management",
    "Fertilizers": "fertilizer, nutrients, manure, chemical treatment",
    "Diseases": "plant diseases, yellow leaves, fungus, infection",
    "Weed Management": "weeds, invasive plants, grass, weed removal",
    "Ambient Conditions": "weather, temperature, cold, heat, climate",
    "General": "general agricultural questions"
}

if not df_csv.empty and "Domain" in df_csv.columns and "Keywords" in df_csv.columns:
    for _, row in df_csv.iterrows():
        AGRI_DOMAINS[row["Domain"]] = str(row["Keywords"])

domain_names = list(AGRI_DOMAINS.keys())
domain_embeddings = embedding_model.encode(list(AGRI_DOMAINS.values()), convert_to_tensor=True)

# ---------------------------
# Domain Styles (light pastel colors + icons)
# ---------------------------
domain_styles = {
    "Soil": {"color": "#F4E1C1", "icon": "🪴"},
    "Seed Quality": {"color": "#FFFACD", "icon": "🌱"},
    "Irrigation": {"color": "#D6F5FF", "icon": "💧"},
    "Pests": {"color": "#FFD6CC", "icon": "🐛"},
    "Fertilizers": {"color": "#DFFFD6", "icon": "🧪"},
    "Diseases": {"color": "#FFE4E1", "icon": "🦠"},
    "Weed Management": {"color": "#E5FFD5", "icon": "🌿"},
    "Ambient Conditions": {"color": "#E0F7FA", "icon": "☀"},
    "General": {"color": "#F5F5F5", "icon": "❓"}
}

# ---------------------------
# Helper Functions
# ---------------------------
def translate_text(text: str, src="auto", dest="en") -> str:
    try:
        return GoogleTranslator(source=src, target=dest).translate(text)
    except:
        return text

def classify_domain(question: str) -> str:
    q_emb = embedding_model.encode(question, convert_to_tensor=True)
    similarities = util.cos_sim(q_emb, domain_embeddings)[0]
    best_idx = similarities.argmax()
    return domain_names[best_idx]

def serpapi_search(query: str, num=5):
    if not SERPAPI_KEY:
        return []
    try:
        res = GoogleSearch({"engine":"google","q":query,"num":num,"api_key":SERPAPI_KEY}).get_dict()
        organic = res.get("organic_results", []) or res.get("organic", [])
        results = [{"title": r.get("title",""),"link":r.get("link") or r.get("url"),"snippet":r.get("snippet","")}
                   for r in organic if r.get("link") or r.get("url")]
        return results[:num]
    except:
        return []

def build_summary_from_snippets(snippets: List[str]) -> str:
    joined = " ".join([s for s in snippets if s])
    if not joined.strip(): return "No concise summary found. Check links below."
    parts = re.split(r'(?<=[.!?])\s+', joined)
    return " ".join(parts[:3])

def synthesize_polly_b64(text: str, lang_code="en") -> str:
    if not polly_client: return None
    voice = polly_voices.get(lang_code, "Aditi")
    try:
        resp = polly_client.synthesize_speech(Text=text, OutputFormat="mp3", VoiceId=voice)
        return base64.b64encode(resp["AudioStream"].read()).decode("utf-8")
    except:
        return None

def play_hidden_audio_b64(b64_audio: str):
    if not b64_audio: return
    st.markdown(f"""
    <audio autoplay hidden>
      <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
    </audio>
    """, unsafe_allow_html=True)

# ---------------------------
# CSS (light, clean UI)
# ---------------------------
st.markdown("""
<style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #31373E;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Dropdown select box */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #40464F !important;
        color: white !important;
        border-radius: 8px;
        border: 1px solid #5A5F66 !important;
    }

    /* Dropdown menu options */
    div[data-baseweb="popover"] {
        background-color: #40464F !important;
        color: white !important;
    }

    /* Option text inside dropdown */
    div[data-baseweb="option"] {
        color: white !important;
    }

    /* Success message (CSV loaded) */
    .stSuccess {
        background-color: #2E3B32 !important;
        color: #C8FACC !important;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Streamlit UI
# ---------------------------
st.markdown("<h1 style='text-align:center;color:#3D9970'>🌾 FarmQ Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;font-size:16px'>Get clear, domain-specific agricultural insights with translation, web results, and voice output.</p>", unsafe_allow_html=True)

user_q = st.text_area(f"Your Question ({selected_lang_name})", height=150)

if st.button("🔍 Get Solution") and user_q.strip():
    with st.spinner("Processing..."):
        translated = translate_text(user_q, src=selected_lang_code, dest="en")
        if translated.lower() != user_q.lower():
            st.info(f"Translated: {translated}")

        domain = classify_domain(translated)
        style = domain_styles.get(domain, {"color":"#f9f9f9","icon":"❓"})
        st.markdown(f"<div class='domain-box' style='background-color:{style['color']}'>{style['icon']} Predicted Domain: {domain}</div>", unsafe_allow_html=True)

        boosters = "soil fertility OR pest management OR irrigation OR fertilizer OR disease management"
        results = serpapi_search(f"{translated} {domain} {boosters}", num=5)
        if not results:
            results = serpapi_search(f"{translated} agriculture {domain}", num=5)

        summary_en = build_summary_from_snippets([r.get("snippet","") for r in results])
        summary_user_lang = translate_text(summary_en, src="en", dest=selected_lang_code)
        st.markdown(f"<div class='summary-box'>{summary_user_lang}</div>", unsafe_allow_html=True)

        st.markdown("### 🔗 Useful Links")
        for r in results:
            st.markdown(f"""
            <div class='card' style='border-left:5px solid {style['color']}'>
                <h4>{style['icon']} {r['title']}</h4>
                <p>{r.get('snippet','')}</p>
                <a href='{r['link']}' target='_blank'>Go to link</a>
            </div>
            """, unsafe_allow_html=True)

        if polly_client:
            speak_text = f"{summary_user_lang} For more details, check the links above."
            audio_b64 = synthesize_polly_b64(speak_text, lang_code=selected_lang_code)
            if audio_b64: play_hidden_audio_b64(audio_b64)