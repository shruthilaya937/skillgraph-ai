import streamlit as st
import PyPDF2
import string
import plotly.graph_objects as go
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SkillGraph AI",
    page_icon="🤖",
    layout="wide"
)

# ---------------- DARK STYLE ----------------
st.markdown("""
<style>
.big-title {
    font-size: 42px;
    font-weight: bold;
    color: #00F5D4;
}
.subtitle {
    color: #9CA3AF;
    margin-bottom: 20px;
}
.legend-card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #00F5D4;
}
.section-card {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 12px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>🤖 SkillGraph AI Resume Ranking System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered resume screening using TF-IDF and Cosine Similarity</div>", unsafe_allow_html=True)

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    filtered_words = [
        word for word in words
        if word.isalpha() and word not in stopwords.words('english')
    ]
    return " ".join(filtered_words)

# ---------------- CLASSIFICATION FUNCTION ----------------
def classify(score):
    percent = score * 100

    if percent < 10:
        return "Very Low"
    elif percent < 20:
        return "Low"
    elif percent < 35:
        return "Moderate"
    elif percent < 50:
        return "Strong"
    else:
        return "Excellent"


# ---------------- INPUT SECTION ----------------
job_description = st.text_area("📌 Paste Job Description Here", height=200)

uploaded_files = st.file_uploader(
    "📄 Upload Multiple Resumes (PDF Only)",
    type=["pdf"],
    accept_multiple_files=True
)

st.write("")

# ---------------- ANALYZE ----------------
if st.button("🚀 Analyze Resumes"):

    if not job_description or not uploaded_files:
        st.warning("Please paste a job description and upload at least one resume.")
    else:

        cleaned_job = clean_text(job_description)

        resume_texts = []
        resume_names = []

        for uploaded_file in uploaded_files:
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted

            cleaned_resume = clean_text(text)
            resume_texts.append(cleaned_resume)
            resume_names.append(uploaded_file.name)

        documents = [cleaned_job] + resume_texts

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)

        job_vector = tfidf_matrix[0]
        resume_vectors = tfidf_matrix[1:]

        similarities = cosine_similarity(job_vector, resume_vectors)
        scores = similarities.flatten()

        ranked_resumes = sorted(
            zip(resume_names, scores),
            key=lambda x: x[1],
            reverse=True
        )

        st.subheader("📊 Resume Match Analysis")

        # ----------- Layout with Columns -----------
        col1, col2 = st.columns([3, 1])

        # ----------- Graph (ALL RESUMES) -----------
        with col1:

            names = [item[0] for item in ranked_resumes]
            scores_percent = [round(item[1] * 100, 2) for item in ranked_resumes]

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=scores_percent[::-1],
                y=names[::-1],
                orientation='h',
                text=[f"{score}%" for score in scores_percent[::-1]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Match: %{x}%<extra></extra>",
                marker=dict(line=dict(width=1))
            ))

            fig.update_layout(
                template="plotly_dark",
                title="All Resume Match Percentages",
                xaxis_title="Match Percentage",
                yaxis_title="Resumes",
                height=650,
                margin=dict(l=20, r=20, t=60, b=20),
                bargap=0.3
            )

            fig.update_xaxes(range=[0, 65])   # FIXED RANGE 0–65

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={'displayModeBar': False}
            )

        # ----------- Cosine Legend (Right Side) -----------
        with col2:
            st.markdown("""
            <div class="legend-card">
            <h4>📘 Similarity Guide</h4>
            🔴 0–10% → Very Low Match<br><br>
            🟠 10–20% → Low Match<br><br>
            🟡 20–35% → Moderate Match<br><br>
            🟢 35–50% → Strong Match<br><br>
            🔵 50%+ → Excellent Match
            </div>
            """, unsafe_allow_html=True)

        # ----------- SHORTLISTING  -----------
        shortlisted = []

        for name, score in ranked_resumes:
            category = classify(score)
            percent = round(score * 100, 2)

            if category in ["Strong", "Moderate", "Excellent"]:
                shortlisted.append((name, percent, category))

        # ----------- Shortlisted Section -----------
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("🟢 Shortlisted Candidates ")

        if shortlisted:
            for name, percent, category in shortlisted:
                st.write(f"✔ {name} → {percent}% ({category})")
        else:
            st.write("No candidates shortlisted.")


        st.markdown("</div>", unsafe_allow_html=True)
