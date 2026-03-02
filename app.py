import streamlit as st
import PyPDF2
import string
import nltk
import pandas as pd
import io

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SkillGraph AI",
    page_icon="🤖",
    layout="wide"
)

# ---------------- PREMIUM DARK THEME ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.big-title {
    font-size: 48px;
    font-weight: 800;
    background: linear-gradient(90deg, #00F5D4, #00BBF9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

.subtitle {
    color: #cfd8dc;
    font-size: 18px;
    margin-bottom: 30px;
}

.legend-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 18px;
    border: 1px solid rgba(0,245,212,0.4);
    box-shadow: 0 0 20px rgba(0,245,212,0.2);
}

.section-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    padding: 25px;
    border-radius: 20px;
    margin-top: 25px;
    border: 1px solid rgba(0,187,249,0.3);
    box-shadow: 0 0 25px rgba(0,187,249,0.2);
}

.stButton>button {
    background: linear-gradient(90deg, #00F5D4, #00BBF9);
    color: black;
    font-weight: 600;
    border-radius: 12px;
    padding: 10px 25px;
}

.stDownloadButton>button {
    background: linear-gradient(90deg, #00F5D4, #00BBF9);
    color: black;
    font-weight: 600;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>SkillGraph AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Premium Semantic Resume Screening System</div>", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

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

# ---------------- CLASSIFICATION ----------------
def classify(score):
    percent = score * 100
    if percent < 40:
        return "Low"
    elif percent < 55:
        return "Moderate"
    elif percent < 70:
        return "Strong"
    else:
        return "Excellent"

# ---------------- TECH FILTER ----------------
NON_TECH_WORDS = {
    "skills","engineering","expertise","knowledge",
    "experience","software","development","engineer",
    "project","team","work","using","system"
}

def extract_common_skills(job_text, resume_text, top_n=8):
    job_words = set(job_text.split())
    resume_words = set(resume_text.split())
    common = job_words.intersection(resume_words)

    technical = [
        word for word in common
        if len(word) > 3 and word not in NON_TECH_WORDS
    ]

    return sorted(technical)[:top_n]

# ---------------- INPUT ----------------
job_description = st.text_area("📌 Paste Job Description", height=200)

uploaded_files = st.file_uploader(
    "📄 Upload Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

st.write("")

# ---------------- ANALYZE ----------------
if st.button("🚀 Analyze Resumes"):

    if not job_description or not uploaded_files:
        st.warning("Please provide job description and resumes.")
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

        job_embedding = model.encode(cleaned_job, convert_to_tensor=True)
        resume_embeddings = model.encode(resume_texts, convert_to_tensor=True)

        similarities = util.cos_sim(job_embedding, resume_embeddings)[0]
        scores = similarities.cpu().numpy()

        ranked_resumes = sorted(
            zip(resume_names, resume_texts, scores),
            key=lambda x: x[2],
            reverse=True
        )

        col1, col2 = st.columns([3,1])

        # -------- Graph --------
        with col1:
            names = [item[0] for item in ranked_resumes]
            scores_percent = [round(item[2]*100,2) for item in ranked_resumes]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=scores_percent[::-1],
                y=names[::-1],
                orientation='h',
                text=[f"{score}%" for score in scores_percent[::-1]],
                textposition="outside"
            ))

            fig.update_layout(template="plotly_dark", height=650)
            fig.update_xaxes(range=[0,100])
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # -------- Guide Box --------
        with col2:
            st.markdown("""
            <div class="legend-card">
            <h4>Similarity Guide</h4>
            🔴 0–40% → Low<br><br>
            🟡 40–55% → Moderate<br><br>
            🟢 55–70% → Strong<br><br>
            🔵 70%+ → Excellent
            </div>
            """, unsafe_allow_html=True)

        # -------- Shortlist --------
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Shortlisted Candidates")

        shortlisted_data = []

        for name,resume_text,score in ranked_resumes:

            category = classify(score)
            percent = round(score*100,2)

            if category in ["Moderate","Strong","Excellent"]:

                matched_skills = extract_common_skills(cleaned_job,resume_text)

                st.write(f"✔ {name} — {percent}% ({category})")
                st.write(f"Technical Skills: {', '.join(matched_skills)}")
                st.write("")

                shortlisted_data.append({
                    "Resume Name": name,
                    "Match %": percent,
                    "Category": category,
                    "Technical Skills": ", ".join(matched_skills)
                })

        st.markdown("</div>", unsafe_allow_html=True)

        # -------- PDF DOWNLOAD --------
        if shortlisted_data:

            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer)
            elements = []
            styles = getSampleStyleSheet()

            elements.append(Paragraph("<b>SkillGraph AI - Shortlisted Report</b>", styles['Heading1']))
            elements.append(Spacer(1, 0.3 * inch))

            for item in shortlisted_data:
                elements.append(Paragraph(
                    f"<b>{item['Resume Name']}</b> — {item['Match %']}% ({item['Category']})",
                    styles['Normal']
                ))
                elements.append(Paragraph(
                    f"Technical Skills: {item['Technical Skills']}",
                    styles['Normal']
                ))
                elements.append(Spacer(1, 0.3 * inch))

            doc.build(elements)
            buffer.seek(0)

            st.download_button(
                label="📄 Download Shortlisted Report (PDF)",
                data=buffer,
                file_name="shortlisted_candidates.pdf",
                mime="application/pdf"
            )
