import os
import base64
import re
import pandas as pd
import streamlit as st
import speech_recognition as sr
from deep_translator import GoogleTranslator
from serpapi import GoogleSearch
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from gtts import gTTS

load_dotenv()

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="🌾 FarmQ Assistant", layout="centered")

# ---------------------------
# ENV VARIABLES
# ---------------------------
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# ---------------------------
# LANGUAGE SETTINGS
# ---------------------------
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Bengali": "bn"
}

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("🌐 Language Settings")

input_lang = st.sidebar.selectbox("Input Language", LANGUAGES.keys())
output_lang = st.sidebar.selectbox("Output Language", LANGUAGES.keys())

in_lang_code = LANGUAGES[input_lang]
out_lang_code = LANGUAGES[output_lang]

# ---------------------------
# LOAD NLP MODEL
# ---------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_model()

# ---------------------------
# AGRI DOMAINS
# ---------------------------
AGRI_DOMAINS = {
    "Soil": "soil fertility nutrients",
    "Seed Quality": "seed germination quality",
    "Irrigation": "water irrigation drought",
    "Pests": "pests insects crop damage",
    "Fertilizers": "fertilizer nutrients",
    "Diseases": "plant diseases fungus",
    "Weed Management": "weeds control",
    "Ambient Conditions": "weather temperature climate",
    "General": "general agriculture"
}

domain_names = list(AGRI_DOMAINS.keys())
domain_embeddings = embedding_model.encode(
    list(AGRI_DOMAINS.values()), convert_to_tensor=True
)

# ---------------------------
# VOICE TO TEXT
# ---------------------------
def voice_to_text(lang):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙 Speak now...")
        audio = r.listen(source, timeout=6, phrase_time_limit=8)

    try:
        return r.recognize_google(audio, language=lang)
    except:
        st.error("❌ Could not recognize speech")
        return ""

# ---------------------------
# TRANSLATION
# ---------------------------
def translate_text(text, src, dest):
    try:
        return GoogleTranslator(source=src, target=dest).translate(text)
    except:
        return text

# ---------------------------
# DOMAIN CLASSIFICATION
# ---------------------------
def classify_domain(text):
    emb = embedding_model.encode(text, convert_to_tensor=True)
    sims = util.cos_sim(emb, domain_embeddings)[0]
    return domain_names[sims.argmax()]

# ---------------------------
# SERP SEARCH
# ---------------------------
def serp_search(query):
    if not SERPAPI_KEY:
        return []
    res = GoogleSearch({
        "engine": "google",
        "q": query,
        "num": 5,
        "api_key": SERPAPI_KEY
    }).get_dict()
    return res.get("organic_results", [])

# ---------------------------
# SUMMARY
# ---------------------------
def summarize(snippets):
    text = " ".join(snippets)
    sents = re.split(r'(?<=[.!?])\s+', text)
    return " ".join(sents[:3])

# ---------------------------
# TEXT TO SPEECH (gTTS – FIXED)
# ---------------------------
def speak(text, lang):
    try:
        tts = gTTS(text=text, lang=lang)
        file = "output.mp3"
        tts.save(file)

        with open(file, "rb") as f:
            audio = f.read()

        return base64.b64encode(audio).decode()
    except Exception as e:
        st.error(f"Voice error: {e}")
        return None

# ---------------------------
# UI
# ---------------------------
st.title("🌾 FarmQ Assistant")
st.caption("🎙 Voice Input → 🌐 Translation → 🔊 Voice Output")

# VOICE INPUT
if st.button("🎤 Ask by Voice"):
    st.session_state["question"] = voice_to_text(in_lang_code)

# TEXT INPUT
user_q = st.text_area(
    "Your Question",
    value=st.session_state.get("question", ""),
    height=120
)

# PROCESS
if st.button("🔍 Get Solution") and user_q.strip():
    with st.spinner("Processing..."):

        # Translate input → English
        q_en = translate_text(user_q, in_lang_code, "en")

        # Domain detection
        domain = classify_domain(q_en)

        # Search
        results = serp_search(q_en + " agriculture")
        snippets = [r.get("snippet", "") for r in results]

        # Summary
        summary_en = summarize(snippets)

        # Translate output
        summary_out = translate_text(summary_en, "en", out_lang_code)

        # OUTPUT
        st.success(f"📌 Domain: {domain}")
        st.markdown(f"### 📝 Answer ({output_lang})")
        st.write(summary_out)

        st.markdown("### 🔗 Useful Links")
        for r in results:
            st.markdown(f"- [{r.get('title')}]({r.get('link')})")

        # VOICE OUTPUT
        audio = speak(summary_out, out_lang_code)
        if audio:
            st.markdown(f"""
            <audio autoplay controls>
                <source src="data:audio/mp3;base64,{audio}">
            </audio>
            """, unsafe_allow_html=True)
