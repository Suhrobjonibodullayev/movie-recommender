<div align="center">

# 🎬 CineMatch AI — Intelligent Movie Recommender System

An end-to-end Content-Based Movie Recommendation Engine built with **Scikit-learn**, **NLP**, and **Streamlit**.  
Dynamically models multi-item user preference vectors in real-time to discover cinema tailored to your taste.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Live Demo](#-https://cinematchaii.streamlit.app) • [System Architecture](#-system-architecture--methodology) • [Getting Started](#-getting-started) • [Engineering Highlights](#-engineering-highlights-memory--performance)

</div>

---

## 📌 Overview

Traditional single-item recommenders fail when a user enjoys diverse genres (e.g., both sci-fi mind-benders and classic mafia dramas). **CineMatch AI** addresses this by accepting a compound seed selection ($\ge 3$ movies), aggregating their latent semantic representations into a unified **User Profile Vector**, and performing instantaneous cosine similarity retrieval across 4,800+ films.

### 🌟 Key Features
- **Curated Diversity Starter Pool:** 10 carefully selected titles spanning distinct genres to solve the cold-start problem.
- **Dynamic User Profiling:** Aggregates embeddings across multiple user picks rather than computing single-item similarity.
- **Real-Time Match Scoring:** Calculates percentage match using normalized cosine distance metrics.
- **Dynamic Poster Fetching:** Integrated with TMDb API to dynamically fetch 500px high-resolution posters with fallback handling.
- **Zero-Latency In-Memory Inference:** Optimized sparse matrices ensure retrieval times under **5 ms**.

---

## 🧠 System Architecture & Methodology

```text
[ Raw TMDb Dataset ]
        │
        ▼ (Extraction & Parsing)
[ Feature Extraction: Overview, Genres, Director, Top 3 Cast, Keywords ]
        │
        ▼ (Text Preprocessing)
[ String Sanitization (Token Merging) + Porter Stemming ]
        │
        ▼ (Vectorization)
[ CountVectorizer (5,000 Dimensions) -> Compressed Sparse Matrix ]
        │
        ▼ (User Interaction)
[ 3+ Seed Movies Selected ] ──► [ User Profile Mean Vector Aggregation ]
                                                │
                                                ▼ (Inference)
                                    [ Cosine Similarity Computation ]
                                                │
                                                ▼
                                    [ Top-N Filtered Recommendations ]
```

### 1. Feature Engineering & "Metadata Soup"
To capture narrative nuances and director/actor styles, metadata fields are transformed into a single token soup:
- **Director & Cast:** Spaces are stripped (e.g., `Christopher Nolan` $\rightarrow$ `christophernolan`) to prevent generic first-name matches.
- **Overview & Keywords:** Tokenized, stop words removed, and stemmed via NLTK's `PorterStemmer`.

### 2. Mathematical Formulation
Given a set of user-selected movie indices $I = \{i_1, i_2, \dots, i_k\}$ where $k \ge 3$, the aggregate user preference vector $\mathbf{v}_{\text{user}}$ is computed as:

$$\mathbf{v}_{\text{user}} = \frac{1}{k} \sum_{i \in I} \mathbf{v}_i$$

The match score against candidate movie vector $\mathbf{v}_m$ is evaluated via Cosine Similarity:

$$\text{Sim}(\mathbf{v}_{\text{user}}, \mathbf{v}_m) = \frac{\mathbf{v}_{\text{user}} \cdot \mathbf{v}_m}{\|\mathbf{v}_{\text{user}}\| \|\mathbf{v}_m\|}$$

Seeds in set $I$ are masked out ($S_i = -1.0$) to prevent trivial self-recommendations.

---

## ⚡ Engineering Highlights (Memory & Performance)

| Metric | Dense Matrix Approach | CineMatch AI (Sparse Approach) | Improvement |
| :--- | :--- | :--- | :--- |
| **Model Disk Size** | ~185 MB (`similarity.pkl`) | **~3.1 MB** (`vectors.pkl`) | **98.3% Reduction** |
| **GitHub LFS Needed?** | Yes (Exceeds limits) | **No** (Direct git push) | Frictionless CI/CD |
| **Inference Latency** | Instantaneous lookup | **< 5 ms** (On-the-fly sparse dot product) | Zero perceptible delay |
| **RAM Footprint** | ~250 MB | **< 35 MB** | Cost-effective deployment |

---

## 📂 Repository Structure

```text
.
├── app.py                  # Main Streamlit application
├── generate_models.py      # Data cleaning & model generation pipeline
├── movies_list.pkl         # Lightweight metadata dictionary
├── vectors.pkl             # Compressed sparse matrix (CountVectorizer output)
├── requirements.txt        # Production dependencies
└── README.md               # Technical documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Git

### Local Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/USERNAME/movie-recommender-system.git
   cd movie-recommender-system
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate models (if running locally for the first time):**
   ```bash
   python generate_models.py
   ```

5. **Launch the application:**
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## 🛠️ Tech Stack

- **Frontend & App Framework:** [Streamlit](https://streamlit.io/)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning & NLP:** [Scikit-learn](https://scikit-learn.org/), [NLTK](https://www.nltk.org/), [SciPy](https://scipy.org/)
- **External APIs:** [The Movie Database (TMDb) API](https://www.themoviedb.org/documentation/api)

---

## 📬 Contact & Connect

- **Author:** [Suhrobjon Ibodullayev](https://github.com/USERNAME)
- **GitHub:** [@USERNAME](https://github.com/USERNAME)  
- **LinkedIn:** [linkedin.com/in/USERNAME](https://linkedin.com/in/USERNAME)  
- **Email:** [your_email@gmail.com](mailto:your_email@gmail.com)  

---
<div align="center">
⭐ If you found this project helpful, please consider giving it a star on GitHub!
</div>
