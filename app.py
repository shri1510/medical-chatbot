import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

@st.cache_resource
def load_model_and_data():
    df = pd.read_csv("symptom_department.csv")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['Symptom Description'].tolist(), convert_to_tensor=True)
    return df, model, embeddings

df, model, symptom_embeddings = load_model_and_data()
body_parts = ["head", "chest", "stomach", "legs", "arms", "back", "eyes", "throat", "feet", "hands", "skin", "neck"]

# Session memory
if "messages" not in st.session_state:
    st.session_state.messages = [("assistant", "👋 Hi! I'm your medical assistant. What symptom are you facing today?")]
    st.session_state.symptom_info = {"symptom": "", "location": "", "duration": "", "severity": ""}
    st.session_state.step = 0
    st.session_state.last_input = ""

if st.button("🔄 Start Over"):
    st.session_state.messages = [("assistant", "👋 Hi! I'm your medical assistant. What symptom are you facing today?")]
    st.session_state.symptom_info = {"symptom": "", "location": "", "duration": "", "severity": ""}
    st.session_state.step = 0
    st.session_state.last_input = ""
    st.experimental_rerun()

st.title("🩺 Medical Chatbot")

for role, msg in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(msg)

def retrieve_department(user_input):
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    similarity = util.cos_sim(user_embedding, symptom_embeddings)
    idx = int(similarity.argmax())
    return df.iloc[idx]['Department']

if prompt := st.chat_input("Describe your symptoms..."):
    user_input = prompt.strip().lower()
    st.session_state.messages.append(("user", prompt))

    # Prevent repeated input from advancing flow
    if user_input == st.session_state.last_input:
        response = "You've already mentioned that. Could you provide more detail?"
        st.session_state.messages.append(("assistant", response))
        st.stop()

    st.session_state.last_input = user_input
    info = st.session_state.symptom_info
    step = st.session_state.step

    greetings = ["hi", "hello", "hey"]
    if user_input in greetings:
        response = "Hello! Please describe the symptom you're experiencing."
        st.session_state.messages.append(("assistant", response))
        st.stop()

    if step == 0:
        info["symptom"] = prompt
        found_part = next((bp for bp in body_parts if bp in user_input), None)
        if found_part:
            info["location"] = found_part
            response = "How long have you been experiencing this?"
            st.session_state.step = 2  # Skip step 1
        else:
            response = "Where on your body is this symptom located?"
            st.session_state.step = 1

    elif step == 1:
        info["location"] = prompt
        response = "How long have you been experiencing this?"
        st.session_state.step = 2

    elif step == 2:
        if any(x in user_input for x in ["day", "week", "hour", "month"]):
            info["duration"] = prompt
            response = "How would you rate it? (Mild / Moderate / Severe)"
            st.session_state.step = 3
        else:
            response = "Please mention how long (e.g., '3 days', 'a week', 'an hour')."

    elif step == 3:
        if user_input in ["mild", "moderate", "severe"]:
            info["severity"] = prompt
            full_input = f"{info['symptom']} {info['location']} {info['duration']} {info['severity']}"
            dept = retrieve_department(full_input)
            response = f"✅ Based on the symptoms you shared, I recommend visiting the **{dept}** department."
            st.session_state.symptom_info = {"symptom": "", "location": "", "duration": "", "severity": ""}
            st.session_state.step = 0
            st.session_state.last_input = ""
        else:
            response = "Please answer with: Mild, Moderate, or Severe."

    st.session_state.messages.append(("assistant", response))
