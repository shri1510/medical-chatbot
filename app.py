import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load data & model only once
@st.cache_resource
def load_model_and_data():
    df = pd.read_csv("symptom_department.csv")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['Symptom Description'].tolist(), convert_to_tensor=True)
    return df, model, embeddings

df, model, symptom_embeddings = load_model_and_data()

# Store chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'symptom_info' not in st.session_state:
    st.session_state.symptom_info = {"symptom": "", "location": "", "duration": "", "severity": ""}
if 'step' not in st.session_state:
    st.session_state.step = 0

st.title("🩺 Medical Chatbot")

def retrieve_department(user_input):
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    similarity = util.cos_sim(user_embedding, symptom_embeddings)
    idx = int(similarity.argmax())
    return df.iloc[idx]['Department']

# Display chat history
for msg in st.session_state.messages:
    role, content = msg
    with st.chat_message(role):
        st.markdown(content)

# Input box
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append(("user", prompt))
    response = ""

    if st.session_state.step == 0:
        st.session_state.symptom_info["symptom"] = prompt
        response = "Got it. Where do you feel this symptom?"
        st.session_state.step += 1

    elif st.session_state.step == 1:
        st.session_state.symptom_info["location"] = prompt
        response = "How long have you had this symptom?"
        st.session_state.step += 1

    elif st.session_state.step == 2:
        st.session_state.symptom_info["duration"] = prompt
        response = "On a scale of Mild, Moderate, Severe — how bad is it?"
        st.session_state.step += 1

    elif st.session_state.step == 3:
        st.session_state.symptom_info["severity"] = prompt
        info = st.session_state.symptom_info
        full_input = f"{info['symptom']} {info['location']} {info['duration']} {info['severity']}"
        department = retrieve_department(full_input)
        response = f"Based on that, I recommend visiting the **{department}** department. ✅"
        st.session_state.step = 0  # Reset

    st.session_state.messages.append(("assistant", response))
    with st.chat_message("assistant"):
        st.markdown(response)
