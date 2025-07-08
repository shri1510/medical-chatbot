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

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = [("assistant", "👋 Hi! I'm your medical assistant. What symptom are you facing today?")]
    st.session_state.symptom_info = {"symptom": "", "location": "", "duration": "", "severity": ""}
    st.session_state.step = 0
    st.session_state.last_input = ""

st.title("🩺 Medical Chatbot")

# Restart button
if st.button("🔄 Start Over"):
    st.session_state.messages = [("assistant", "👋 Hi! I'm your medical assistant. What symptom are you facing today?")]
    st.session_state.symptom_info = {"symptom": "", "location": "", "duration": "", "severity": ""}
    st.session_state.step = 0
    st.session_state.last_input = ""
    st.experimental_rerun()

# Display chat history
for role, msg in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(msg)

# Helper: predict department
def retrieve_department(user_input):
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    similarity = util.cos_sim(user_embedding, symptom_embeddings)
    idx = int(similarity.argmax())
    return df.iloc[idx]['Department']

# ✅ Input and logic block
prompt = st.chat_input("Describe your symptoms...")
if prompt:
    user_input = prompt.strip().lower()
    st.session_state.messages.append(("user", prompt))

    # Prevent accidental repeated inputs
    if user_input == st.session_state.last_input:
        st.session_state.messages.append(("assistant", "You already said that — could you provide something new?"))
    else:
        st.session_state.last_input = user_input
        info = st.session_state.symptom_info
        step = st.session_state.step

        greetings = ["hi", "hello", "hey"]
        if user_input in greetings:
            st.session_state.messages.append(("assistant", "Hello! Please tell me your symptoms."))
        elif step == 0:
            info["symptom"] = prompt
            found = next((bp for bp in body_parts if bp in user_input), None)
            if found:
                info["location"] = found
                st.session_state.step = 2
                st.session_state.messages.append(("assistant", "How long have you been experiencing this?"))
            else:
                st.session_state.step = 1
                st.session_state.messages.append(("assistant", "Where on your body are you feeling this?"))
        elif step == 1:
            info["location"] = prompt
            st.session_state.step = 2
            st.session_state.messages.append(("assistant", "How long have you been experiencing this?"))
        elif step == 2:
            if any(t in user_input for t in ["day", "week", "hour", "month"]):
                info["duration"] = prompt
                st.session_state.step = 3
                st.session_state.messages.append(("assistant", "How would you rate it? (Mild / Moderate / Severe)"))
            else:
                st.session_state.messages.append(("assistant", "Please mention how long — e.g., '3 days', 'a week'."))
        elif step == 3:
            if user_input in ["mild", "moderate", "severe"]:
                info["severity"] = prompt
                full_input = f"{info['symptom']} {info['location']} {info['duration']} {info['severity']}"
                dept = retrieve_department(full_input)
                st.session_state.messages.append(("assistant", f"✅ Based on what you've told me, you should consult the **{dept}** department."))
                st.session_state.step = 0
                st.session_state.symptom_info = {"symptom": "", "location": "", "duration": "", "severity": ""}
                st.session_state.last_input = ""
            else:
                st.session_state.messages.append(("assistant", "Please answer with: Mild, Moderate, or Severe."))
