import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load model + data
@st.cache_resource
def load_model_and_data():
    df = pd.read_csv("symptom_department.csv")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['Symptom Description'].tolist(), convert_to_tensor=True)
    return df, model, embeddings

df, model, symptom_embeddings = load_model_and_data()
body_parts = ["head", "chest", "stomach", "legs", "arms", "back", "eyes", "throat", "feet", "hands", "skin", "neck"]

# Initialize session state
def init_chat():
    st.session_state.messages = [("assistant", "👋 Hi! I'm your medical assistant. Tell me what symptoms you're experiencing.")]
    st.session_state.symptom_info = {"symptom": "", "location": "", "duration": "", "severity": ""}
    st.session_state.step = 0
    st.session_state.last_input = ""

if "messages" not in st.session_state:
    init_chat()

st.title("🩺 Medical Chatbot")

# Restart button
if st.button("🔄 Start Over"):
    init_chat()
    st.experimental_rerun()

# Show previous messages
for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(text)

# RAG match function
def recommend_department(text):
    emb = model.encode(text, convert_to_tensor=True)
    sim = util.cos_sim(emb, symptom_embeddings)
    idx = int(sim.argmax())
    return df.iloc[idx]["Department"]

# Chat input and full logic inside this block
user_prompt = st.chat_input("Describe your symptoms...")

if user_prompt:
    user_input = user_prompt.strip().lower()
    st.session_state.messages.append(("user", user_prompt))

    # Avoid repeat inputs
    if user_input == st.session_state.last_input:
        response = "You've already said that. Could you add more info?"
        st.session_state.messages.append(("assistant", response))
    else:
        st.session_state.last_input = user_input
        step = st.session_state.step
        info = st.session_state.symptom_info

        # Greeting handler
        if user_input in ["hi", "hello", "hey", "good morning", "good evening"]:
            st.session_state.messages.append(("assistant", "Hello! Please describe your symptoms."))
        elif step == 0:
            info["symptom"] = user_prompt
            # Try to detect location early
            part = next((bp for bp in body_parts if bp in user_input), None)
            if part:
                info["location"] = part
                st.session_state.step = 2
                st.session_state.messages.append(("assistant", "How long has this been bothering you?"))
            else:
                st.session_state.step = 1
                st.session_state.messages.append(("assistant", "Where on your body are you feeling this?"))
        elif step == 1:
            info["location"] = user_prompt
            st.session_state.step = 2
            st.session_state.messages.append(("assistant", "How long has this been bothering you?"))
        elif step == 2:
            if any(x in user_input for x in ["day", "week", "hour", "month", "year"]):
                info["duration"] = user_prompt
                st.session_state.step = 3
                st.session_state.messages.append(("assistant", "How severe is it? (Mild / Moderate / Severe)"))
            else:
                st.session_state.messages.append(("assistant", "Please mention how long (e.g., '3 days', 'a week')."))
        elif step == 3:
            if user_input in ["mild", "moderate", "severe"]:
                info["severity"] = user_prompt
                full_text = f"{info['symptom']} {info['location']} {info['duration']} {info['severity']}"
                dept = recommend_department(full_text)
                response = f"✅ Based on what you've told me, I recommend visiting the **{dept}** department."
                st.session_state.messages.append(("assistant", response))
                init_chat()  # reset chat after recommendation
            else:
                st.session_state.messages.append(("assistant", "Please answer with: Mild, Moderate, or Severe."))
