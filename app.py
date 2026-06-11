import streamlit as st
from dotenv import load_dotenv
import os
from google import genai

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime



# ---------------------------------
# Load Environment Variables
# ---------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        st.error("GEMINI_API_KEY not found")
        st.stop()

client = genai.Client(api_key=api_key)

# ---------------------------------
# Load Embedding Model
# ---------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# ---------------------------------
# Session State
# ---------------------------------

if "questions_attempted" not in st.session_state:
    st.session_state.questions_attempted = 0

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------
# Sidebar
# ---------------------------------

st.sidebar.title("🤖 AI Interview Coach")

st.sidebar.info(
    """
    Built using:

    • Python
    • Streamlit
    • Gemini API
    • NLP
    • Sentence Transformers
    """
)

st.sidebar.metric(
    "Questions Attempted",
    st.session_state.questions_attempted
)

# ---------------------------------
# Main UI
# ---------------------------------

st.title("🤖 AI Interview Coach")

st.markdown(
    """
    Practice technical interviews using AI.

    Generate interview questions,
    answer them, and receive
    instant feedback.
    """
)

st.subheader("📊 Dashboard")

if st.session_state.history:

    scores = [
        item["score"]
        for item in st.session_state.history
    ]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Questions Attempted",
        len(scores)
    )

    col2.metric(
        "Average Score",
        round(
            sum(scores)/len(scores),
            2
        )
    )

    col3.metric(
        "Best Score",
        max(scores)
    )

else:

    st.info(
        "No interview history available yet."
    )

# ---------------------------------
# Role Selection
# ---------------------------------

role = st.selectbox(
    "Choose Role",
    [
        "Machine Learning Engineer",
        "Data Scientist",
        "AI Engineer",
        "Python Developer"
    ]
)

# ---------------------------------
# Difficulty Selection
# ---------------------------------

difficulty = st.selectbox(
    "Choose Difficulty",
    [
        "Easy",
        "Medium",
        "Hard"
    ]
)

# ---------------------------------
# Generate Question
# ---------------------------------

if st.button("🎯 Generate Question"):

    try:

        with st.spinner("Generating Question..."):

            prompt = f"""
            Generate ONE {difficulty} interview question
            for a {role}.

            Return only the question.
            """

            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt
            )

            st.session_state.question = response.text
            st.session_state.questions_attempted += 1

    except Exception as e:
        st.error(f"Error generating question: {e}")

# ---------------------------------
# Show Question
# ---------------------------------

if "question" in st.session_state:

    st.subheader("📝 Interview Question")

    st.info(st.session_state.question)

    answer = st.text_area(
        "Type Your Answer Here",
        height=200
    )

    if st.button("📊 Evaluate Answer"):

        if answer.strip() == "":
            st.warning("Please enter an answer.")

        else:

            try:

                # ---------------------------------
                # Gemini Evaluation
                # ---------------------------------

                evaluation_prompt = f"""
                You are a senior technical interviewer.

                Question:
                {st.session_state.question}

                Candidate Answer:
                {answer}

                Evaluate based on:

                1. Technical Accuracy
                2. Completeness
                3. Clarity

                Return in EXACT format:

                Score: X/10

                Strengths:
                - ...

                Weaknesses:
                - ...

                Suggestions:
                - ...
                """

                with st.spinner("Evaluating Answer..."):

                    evaluation = client.models.generate_content(
                        model="gemini-2.5-flash-lite",
                        contents=evaluation_prompt
                    )

                st.subheader("📈 AI Feedback")

                st.write(evaluation.text)

                # ---------------------------------
                # Extract Interview Score
                # ---------------------------------

                score_match = re.search(
                    r"Score:\s*(\d+(\.\d+)?)",
                    evaluation.text
                )

                if score_match:
                    interview_score = float(
                        score_match.group(1)
                    )
                else:
                    interview_score = 0

                # ---------------------------------
                # Generate Ideal Answer
                # ---------------------------------

                ideal_prompt = f"""
                Interview Question:

                {st.session_state.question}

                Provide an ideal interview answer.
                """

                ideal_answer = client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=ideal_prompt
                )

                ideal_text = ideal_answer.text

                # ---------------------------------
                # Similarity Score
                # ---------------------------------

                user_embedding = embedding_model.encode(
                    answer
                )

                ideal_embedding = embedding_model.encode(
                    ideal_text
                )

                similarity = float(
                    cosine_similarity(
                        [user_embedding],
                        [ideal_embedding]
                    )[0][0]
                )

                similarity_percent = round(
                    similarity * 100,
                    2
                )

                if interview_score == 0:

                    interview_score = round(
                        similarity_percent / 10,
                        1
                    )

                # ---------------------------------
                # Save History
                # ---------------------------------

                if (
                    not st.session_state.history
                    or
                    st.session_state.history[-1]["question"]
                    != st.session_state.question
                ):

                    st.session_state.history.append(
                        {
                            "date": datetime.now().strftime(
                                "%Y-%m-%d %H:%M"
                            ),
                            "role": role,
                            "question": st.session_state.question,
                            "score": interview_score,
                            "similarity": similarity_percent
                        }
                    )

                # ---------------------------------
                # Display Results
                # ---------------------------------

                st.subheader(
                    "🧠 NLP Similarity Score"
                )

                progress_value = float(
                    min(
                        similarity_percent / 100,
                        1.0
                    )
                )

                st.progress(progress_value)

                st.metric(
                    "Semantic Similarity",
                    f"{similarity_percent}%"
                )

                st.metric(
                    "Interview Score",
                    f"{interview_score}/10"
                )

                with st.expander(
                    "View Ideal Answer"
                ):
                    st.write(ideal_text)

            except Exception as e:

                st.error(
                    f"Error evaluating answer: {e}"
                )

# ---------------------------------
# Interview History
# ---------------------------------

st.markdown("---")

st.subheader(
    "📚 Interview History"
)

if st.session_state.history:

    for i, item in enumerate(
        reversed(
            st.session_state.history
        ),
        start=1
    ):

        with st.expander(
            f"Question "
            f"{len(st.session_state.history)-i+1}"
            f" | Score "
            f"{item['score']}/10"
        ):

            st.write(
                f"Role: {item['role']}"
            )

            st.write(
                f"Similarity: "
                f"{item['similarity']}%"
            )

            st.write(
                item["question"]
            )

if st.session_state.history:

    scores = [
        item["score"]
        for item in st.session_state.history
    ]

    avg_score = round(
        sum(scores)/len(scores),
        2
    )

    best_score = max(scores)

    report = f"""
AI INTERVIEW REPORT

Role: {role}

Questions Attempted:
{len(scores)}

Average Score:
{avg_score}/10

Best Score:
{best_score}/10

INTERVIEW HISTORY
"""

    for i, item in enumerate(
        st.session_state.history,
        start=1
    ):

        report += f"""

Question {i}

Role:
{item['role']}

Score:
{item['score']}/10

Similarity:
{item['similarity']}%

Question:
{item['question']}
"""

    st.download_button(
        "📄 Download Report",
        report,
        file_name=
        "AI_Interview_Report.txt",
        mime="text/plain"
    )
# ---------------------------------
# Footer
# ---------------------------------

st.markdown("---")

st.caption(
    "Built by Charishma using Python, Streamlit, Gemini and NLP"
)