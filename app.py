import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# 1. SAHIFA SOZLAMALARI VA MAXSUS DIZAYN
# ---------------------------------------------------------
st.set_page_config(
    page_title="CineMatch AI | Film Tavsiya Tizimi",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Zamonaviy interfeys uchun CSS uslublari
st.markdown("""
<style>
    /* Asosiy fon va shriftlar */
    .main {
        background-color: #0e1117;
    }
    
    /* Film kartochkalari */
    .stCard {
        border-radius: 12px;
        padding: 10px;
        background: #1a1c24;
        border: 1px solid #2a2e3d;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .stCard:hover {
        border-color: #ff4b4b;
        transform: translateY(-2px);
    }
    
    /* Moslik foizi nishoni */
    .badge-match {
        background: linear-gradient(135deg, #00c853, #b2ff59);
        color: #000;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 13px;
        display: inline-block;
        margin-bottom: 8px;
    }
    
    /* Muallif bilan bog'lanish bloki (Footer) */
    .footer-box {
        background: #161922;
        border: 1px solid #262a36;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin-top: 50px;
    }
    .contact-link {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #212530;
        color: #ffffff !important;
        text-decoration: none !important;
        padding: 8px 18px;
        border-radius: 8px;
        margin: 6px 8px;
        font-weight: 500;
        border: 1px solid #32384a;
        transition: all 0.2s;
    }
    .contact-link:hover {
        background: #ff4b4b;
        border-color: #ff4b4b;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"

# ---------------------------------------------------------
# 2. XOTIRANI TEJAYDIGAN RESURSLARNI YUKLASH
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_assets():
    """Vektorlar va ma'lumotlarni xotiraga bir marta tez yuklaydi"""
    with open('movies_list.pkl', 'rb') as f:
        movies_dict = pickle.load(f)
    with open('vectors.pkl', 'rb') as f:
        vectors = pickle.load(f)
    
    df = pd.DataFrame(movies_dict)
    return df, vectors

@st.cache_data(show_spinner=False, ttl=7200)
def fetch_poster(movie_id):
    """TMDb API orqali rasmni xavfsiz yuklab oladi"""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
    fallback_poster = "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=500&auto=format&fit=crop"
    try:
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            poster_path = res.json().get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        pass
    return fallback_poster

try:
    df, vectors = load_assets()
except FileNotFoundError:
    st.error("⚠️ `movies_list.pkl` yoki `vectors.pkl` fayli topilmadi. Fayllar shu katalogda ekanligiga ishonch hosil qiling.")
    st.stop()

# ---------------------------------------------------------
# 3. YON PANEL (LOYIHA METADATALARI)
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=500&auto=format&fit=crop", use_container_width=True)
    st.title("🎯 Loyiha Haqida")
    st.markdown("""
    **CineMatch AI** — foydalanuvchining ko'p qirrali ta'bini umumlashtiruvchi tavsiya tizimi.
    
    * **Model:** Bag of Words + CountVectorizer (5000 xususiyat)
    * **Algoritm:** Cosine Similarity
    * **Baza hajmi:** ~4,800 ta film
    * **Optimallashtirish:** Sparse Matrix (`vectors.pkl` hajmi atigi ~3 MB)
    """)
    st.divider()
    st.caption("🚀 Portfolio ML Loyihasi")

# ---------------------------------------------------------
# 4. ASOSIY INTERFEYS VA 10 TA DIVERSITY FILM POOLI
# ---------------------------------------------------------
st.title("🎬 CineMatch AI — Shaxsiy Film Tavsiya Tizimi")
st.markdown("Quyidagi 10 ta durdona filmdan **kamida 3 tasini** belgilang. Tizim ularning umumiy vektorini hisoblab, sizga eng mos keluvchi yangi filmlarni topadi.")

STARTER_TITLES = [
    "Inception",
    "The Dark Knight",
    "Pulp Fiction",
    "Forrest Gump",
    "Spirited Away",
    "The Matrix",
    "Interstellar",
    "The Godfather",
    "Whiplash",
    "Gladiator"
]

starter_movies = df[df['title'].isin(STARTER_TITLES)].drop_duplicates(subset=['title']).head(10).reset_index(drop=True)

# 10 ta filmni 2 qatorda 5 tadan joylashtirish
selected_movie_ids = []
cols = list(st.columns(5)) + list(st.columns(5))

for idx, row in starter_movies.iterrows():
    with cols[idx]:
        poster_url = fetch_poster(row['movie_id'])
        st.image(poster_url, use_container_width=True)
        st.markdown(f"**{row['title']}**")
        st.caption(f"📅 {row['release_year']} | 🎭 {row['genres_display'].split(',')[0]}")
        
        if st.checkbox("Tanlash", key=f"starter_{row['movie_id']}"):
            selected_movie_ids.append(row['movie_id'])

# ---------------------------------------------------------
# 5. PROGRESS VA HISOBLASH BOSQICHI
# ---------------------------------------------------------
st.divider()
selected_count = len(selected_movie_ids)
progress_ratio = min(selected_count / 3.0, 1.0)

col_metric, col_action = st.columns([3, 1])

with col_metric:
    st.progress(progress_ratio)
    if selected_count < 3:
        st.info(f"Hozirda tanlandi: **{selected_count} / 3**. Hisoblash uchun yana kamida **{3 - selected_count} ta** film belgilang.")
    else:
        st.success(f"Ajoyib! **{selected_count} ta film** tanlandi. Tavsiyalar tayyor.")

with col_action:
    btn_ready = st.button("Tavsiya Olish 🚀", type="primary", use_container_width=True, disabled=(selected_count < 3))

# ---------------------------------------------------------
# 6. TAVSIYALAR NATIJASI (REAL-TIME COSINE SIMILARITY)
# ---------------------------------------------------------
if btn_ready:
    with st.spinner("Vektorlar bo'yicha moslik tahlil qilinmoqda..."):
        # 1. Tanlangan filmlar indekslarini aniqlaymiz
        selected_indices = df[df['movie_id'].isin(selected_movie_ids)].index.tolist()

        # 2. Foydalanuvchi profil vektorini hisoblaymiz (siyrak matritsa o'rtachasi)
        user_profile_vec = np.asarray(vectors[selected_indices].mean(axis=0))

        # 3. Bir zumda kosinus o'xshashligini hisoblaymiz
        scores = cosine_similarity(user_profile_vec, vectors).flatten()

        # 4. Foydalanuvchi tanlagan filmlar qayta tavsiya qilinmasligi uchun ularni chetlatamiz
        for idx in selected_indices:
            scores[idx] = -1.0

        # 5. Eng yuqori 3 ta mos film
        top_indices = np.argsort(scores)[::-1][:3]

        st.subheader("✨ Sizning didingizga eng yaqin 3 ta tavsiya:")
        rec_cols = st.columns(3)

        for i, idx in enumerate(top_indices):
            movie = df.iloc[idx]
            match_pct = round(float(scores[idx]) * 100, 1)
            rec_poster = fetch_poster(movie['movie_id'])

            with rec_cols[i]:
                st.markdown(f"<span class='badge-match'>🎯 {match_pct}% Moslik</span>", unsafe_allow_html=True)
                st.image(rec_poster, use_container_width=True)
                st.markdown(f"### {movie['title']} ({movie['release_year']})")
                st.markdown(f"**Rejissyor:** `{movie['director_display']}`")
                st.markdown(f"**Janrlar:** {movie['genres_display']}")
                st.markdown(f"**Bosh rollarda:** {movie['cast_display']}")

# ---------------------------------------------------------
# 7. MUALLIF PROFILI VA ALOQA (FOOTER)
# ---------------------------------------------------------
st.markdown("""
<div class='footer-box'>
    <h3>👨‍💻 Loyiha Muallifi</h3>
    <p style='color: #8b949e; max-width: 600px; margin: 0 auto 15px auto;'>
        Ushbu loyiha Machine Learning va Natural Language Processing (NLP) yondashuvlarini amaliyotda qo'llash maqsadida yaratilgan.
    </p>
    <a href="https://github.com/Suhrobjonibodullayev" target="_blank" class="contact-link">
        🐙 GitHub Profilim
    </a>
    <a href="mailto:suhrobjonibodullaev@gmail.com" class="contact-link">
        ✉️ Men bilan bog'lanish (Gmail)
    </a>
</div>
""", unsafe_allow_html=True)
