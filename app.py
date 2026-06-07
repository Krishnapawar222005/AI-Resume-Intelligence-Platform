import streamlit as st
import pdfplumber
import google.generativeai as genai
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.pdfgen import canvas
from io import BytesIO
genai.configure(
    api_key=st.secrets["GEMINI_API_KEY"]
)

model = genai.GenerativeModel(
      "gemini-2.5-flash"
)

st.set_page_config(
    page_title="AI Resume Intelligence Platform",
    layout="wide"
)

st.title("🚀 AI Resume Intelligence Platform")
st.error("NEW VERSION LOADED")
st.write("Advanced Resume Analysis, ATS Scoring & Job Matching")
st.subheader("🤖 AI TEST")

user_question = st.text_input(
    "Ask anything"
)

st.write("AI SECTION VISIBLE")
SKILLS = [
    "python",
    "java",
    "c++",
    "javascript",
    "react",
    "nodejs",
    "mongodb",
    "sql",
    "html",
    "css",
    "git",
    "github",
    "machine learning",
    "data science",
    "flask",
    "streamlit"
]


def extract_text(pdf_file):
    text = ""

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


def extract_skills(text):
    found_skills = []

    text = text.lower()

    for skill in SKILLS:
        if skill in text:
            found_skills.append(skill)

    return found_skills


def calculate_match(resume_text, jd_text):

    documents = [resume_text, jd_text]

    tfidf = TfidfVectorizer()

    tfidf_matrix = tfidf.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )

    return round(similarity[0][0] * 100, 2)


def generate_pdf_report(
    ats_score,
    match_score,
    found_skills,
    missing_skills,
    recommendations
):

    buffer = BytesIO()

    pdf = canvas.Canvas(buffer)

    pdf.setTitle("Resume Analysis Report")

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(
        50,
        800,
        "AI Resume Intelligence Report"
    )

    pdf.setFont("Helvetica", 12)

    pdf.drawString(
        50,
        770,
        f"ATS Score: {ats_score}%"
    )

    pdf.drawString(
        50,
        750,
        f"Match Score: {match_score}%"
    )

    y = 720

    pdf.drawString(
        50,
        y,
        "Skills Found:"
    )

    y -= 20

    for skill in found_skills:
        pdf.drawString(
            70,
            y,
            f"- {skill}"
        )
        y -= 18

    pdf.drawString(
        50,
        y,
        "Missing Skills:"
    )

    y -= 20

    for skill in missing_skills:
        pdf.drawString(
            70,
            y,
            f"- {skill}"
        )
        y -= 18

    pdf.drawString(
        50,
        y,
        "Recommendations:"
    )

    y -= 20

    for rec in recommendations:
        pdf.drawString(
            70,
            y,
            f"- {rec}"
        )
        y -= 18

    pdf.save()

    buffer.seek(0)

    return buffer


uploaded_resume = st.file_uploader(
    "📄 Upload Resume PDF",
    type=["pdf"]
)

job_description = st.text_area(
    "📋 Paste Job Description Here",
    height=200
)

if uploaded_resume:
    st.write("STEP 1")
    resume_text = extract_text(uploaded_resume)

    found_skills = extract_skills(resume_text)

    ats_score = int(
        (len(found_skills) / len(SKILLS)) * 100
    )

    match_score = 0
    missing_skills = []

    if job_description:

        match_score = calculate_match(
            resume_text,
            job_description
        )

        jd_skills = extract_skills(
            job_description
        )

        for skill in jd_skills:
            if skill not in found_skills:
                missing_skills.append(skill)

    st.subheader("📊 Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "ATS Score",
            f"{ats_score}%"
        )

    with col2:
        st.metric(
            "Match Score",
            f"{match_score}%"
        )

    with col3:
        st.metric(
            "Skills Found",
            len(found_skills)
        )

    st.subheader("📈 Performance Chart")

    chart_data = pd.DataFrame({
        "Metric": [
            "ATS",
            "Match",
            "Skills"
        ],
        "Value": [
            ats_score,
            match_score,
            len(found_skills)
        ]
    })

    fig, ax = plt.subplots(
        figsize=(4, 2)
    )

    ax.bar(
        chart_data["Metric"],
        chart_data["Value"]
    )

    ax.set_ylim(0, 100)
    ax.set_title(
        "Resume Analysis"
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.subheader("📊 ATS Analysis")

    st.progress(ats_score)

    if ats_score >= 80:
        st.success(
            "Excellent ATS Score"
        )
    elif ats_score >= 60:
        st.warning(
            "Good ATS Score - Can Be Improved"
        )
    else:
        st.error(
            "Low ATS Score - Needs Improvement"
        )

    st.subheader("🛠 Skills Found")

    st.write(found_skills)

    if job_description:

        st.subheader(
            "🎯 Resume Match Score"
        )

        st.progress(
            int(match_score)
        )

        st.metric(
            label="Match Score",
            value=f"{match_score}%"
        )

        st.subheader(
            "❌ Missing Skills"
        )

        if missing_skills:

            for skill in missing_skills:
                st.write(
                    "•",
                    skill
                )

        else:

            st.success(
                "No Missing Skills Detected!"
            )   
    st.subheader("🤖 AI Recommendations")

    recommendations = []

    for skill in missing_skills:
        recommendations.append(
            f"Add {skill} to your resume or projects."
        )

    if ats_score < 70:
        recommendations.append(
            "Add more technical skills to improve ATS score."
        )

    if len(found_skills) < 8:
        recommendations.append(
            "Include more projects showcasing your skills."
        )

    if len(recommendations) == 0:
        st.success(
            "Your resume looks strong for this job description."
        )
    else:
        for rec in recommendations:
            st.write("✅", rec)

    st.subheader("💪 Resume Strength Analysis")

    strengths = []
    weaknesses = []

    if ats_score >= 80:
        strengths.append("Strong ATS optimization")
    else:
        weaknesses.append("ATS score can be improved")

    if len(found_skills) >= 10:
        strengths.append("Good technical skill coverage")
    else:
        weaknesses.append("Add more relevant skills")

    if len(missing_skills) == 0:
        strengths.append("Skills match the job description well")
    else:
        weaknesses.append("Some required skills are missing")

    strength_score = min(ats_score + 10, 100)

    st.metric(
        "Resume Strength",
        f"{strength_score}/100"
    )

    st.write("### Strengths")
    for item in strengths:
        st.success(item)

    st.write("### Areas to Improve")
    for item in weaknesses:
        st.warning(item)

    report = generate_pdf_report(
        ats_score,
        match_score,
        found_skills,
        missing_skills,
        recommendations
    )

    st.download_button(
        label="📥 Download PDF Report",
        data=report,
        file_name="resume_analysis_report.pdf",
        mime="application/pdf"
    )

    st.subheader("📄 Resume Content")

    st.text_area(
        "Resume Content",
        resume_text,
        height=300
    )
    st.write("TEST AI SECTION")
    st.subheader("🤖 Resume AI Assistant")

    user_question = st.text_input(
        "Ask anything about your resume"
    )

    if user_question:

        prompt = f"""
        Resume Content:
        {resume_text}

        Job Description:
        {job_description}

        User Question:
        {user_question}

        Answer based on the resume and job description.
        """

        with st.spinner("AI is analyzing..."):

            try:
                response = model.generate_content(
                    prompt
                )

                st.write(response.text)

            except Exception as e:
                st.error(
                    "AI quota exceeded. Please wait 20 seconds and try again."
                )