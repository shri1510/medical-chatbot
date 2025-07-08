import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load model and data once
@st.cache_resource
def load_model_and_data():
    df = pd.read_csv("symptom_department.csv")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["Symptom Description"].tolist(), convert_to_tensor=True)
    return df, model, embeddings

df, model, symptom_embeddings = load_model_and_data()
body_parts = ["head", "chest", "stomach", "legs", "arms", "back", "eyes", "throat", "feet", "hands", "skin", "neck"]

# Init chatbot state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "step" not in st.session_state:
    st.session_state.step = 0
if "symptom_info" not in st.session_state:
    st.session_state.symptom_info = {"symptom": "", "location": "", "duration": "", "severity": ""}

# Greeting at launch
if not st.session_state.messages:
    st.session_state.messages.append(("assistant", "👋 Hi! I'm your medical assistant. What symptoms are you experiencing?"))

st.title("🩺 AI Medical Chatbot")

# Restart conversation
if st.button("🔄 Start Over"):
    st.session_state.messages = [("assistant", "👋 Hi! I'm your medical assistant. What symptoms are you experiencing?")]
    st.session_state.step = 0
    st.session_state.symptom_info = {"symptom": "", "location": "", "duration": "", "severity": ""}
    st.experimental_rerun()

# Show chat history
for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(text)

# RAG logic
def get_department(text):
    user_embedding = model.encode(text, convert_to_tensor=True)
    sim_scores = util.cos_sim(user_embedding, symptom_embeddings)
    best_idx = int(sim_scores.argmax())
    return df.iloc[best_idx]["Department"]

# Actual chatbot logic
prompt = st.chat_input("Type your message...")

if prompt:
    st.session_state.messages.append(("user", prompt))
    msg = prompt.lower().strip()
    step = st.session_state.step
    info = st.session_state.symptom_info

    if msg in ["hi", "hello", "hey"]:
        st.session_state.messages.append(("assistant", "Hello! Please tell me your symptoms."))
    elif step == 0:
        info["symptom"] = prompt
        detected = next((bp for bp in body_parts if bp in msg), None)
        if detected:
            info["location"] = detected
            st.session_state.step = 2
            st.session_state.messages.append(("assistant", "How long have you been experiencing this?"))
        else:
            st.session_state.step = 1
            st.session_state.messages.append(("assistant", "Where in your body are you feeling this?"))
    elif step == 1:
        info["location"] = prompt
        st.session_state.step = 2
        st.session_state.messages.append(("assistant", "How long have you been experiencing this?"))
    elif step == 2:
        if any(x in msg for x in ["day", "week", "hour", "month"]):
            info["duration"] = prompt
            st.session_state.step = 3
            st.session_state.messages.append(("assistant", "How would you rate it? (Mild / Moderate / Severe)"))
        else:
            st.session_state.messages.append(("assistant", "Please enter duration like '3 days', '1 week', etc."))
    elif step == 3:
        if msg in ["mild", "moderate", "severe"]:
            info["severity"] = prompt
            query = f"{info['symptom']} {info['location']} {info['duration']} {info['severity']}"
            dept = get_department(query)
            st.session_state.messages.append(("assistant", f"✅ Based on what you've told me, I recommend the **{dept}** department."))
            # Reset convo
            st.session_state.step = 0
            st.session_state.symptom_info = {"symptom": "", "location": "", "duration": "", "severity": ""}
        else:
            st.session_state.messages.append(("assistant", "Please answer with: Mild, Moderate, or Severe."))
