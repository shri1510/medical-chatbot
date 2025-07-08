import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load data & model
@st.cache_resource
def load_model_and_data():
    df = pd.read_csv("symptom_department.csv")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['Symptom Description'].tolist(), convert_to_tensor=True)
    return df, model, embeddings

df, model, symptom_embeddings = load_model_and_data()

# Chatbot memory
if "messages" not in st.session_state:
    st.session_state.messages = [("assistant", "👋 Hi! I'm your medical assistant. What symptoms are you experiencing today?")]
    st.session_state.symptom_info = {"symptom": "", "location": "", "duration": "", "severity": ""}
    st.session_state.step = 0

# Restart button
if st.button("🔄 Start Over"):
    st.session_state.messages = [("assistant", "👋 Hi! I'm your medical assistant. What symptoms are you experiencing today?")]
    st.session_state.symptom_info = {"symptom": "", "location": "", "duration": "", "severity": ""}
    st.session_state.step = 0
    st.experimental_rerun()

# Chat title
st.title("🩺 Medical Chatbot")

# Display conversation
for role, msg in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(msg)

# RAG function
def retrieve_department(user_input):
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    similarity = util.cos_sim(user_embedding, symptom_embeddings)
    idx = int(similarity.argmax())
    return df.iloc[idx]['Department']

# Chat logic
if prompt := st.chat_input("Describe your symptoms..."):
    st.session_state.messages.append(("user", prompt))

    user_msg = prompt.lower().strip()

    # Greet detection
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    if user_msg in greetings:
        response = "Hello! 👋 Please describe your symptom(s), and I’ll guide you."
        st.session_state.messages.append(("assistant", response))
        st.stop()

    # Dialog step logic
    step = st.session_state.step
    info = st.session_state.symptom_info

    if step == 0:
        info["symptom"] = prompt
        response = "Thanks. Where in your body are you experiencing this?"
        st.session_state.step += 1

    elif step == 1:
        info["location"] = prompt
        response = "How long have you had this symptom?"
        st.session_state.step += 1

    elif step == 2:
        info["duration"] = prompt
        response = "On a scale of Mild, Moderate, Severe — how would you rate it?"
        st.session_state.step += 1

    elif step == 3:
        info["severity"] = prompt
        full_input = f"{info['symptom']} {info['location']} {info['duration']} {info['severity']}"
        department = retrieve_department(full_input)
        response = f"✅ Based on what you've told me, I recommend visiting the **{department}** department."
        st.session_state.step = 0
        st.session_state.symptom_info = {"symptom": "", "location": "", "duration": "", "severity": ""}

    st.session_state.messages.append(("assistant", response))
