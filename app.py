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

# Simple body part list
body_parts = ["head", "chest", "stomach", "legs", "arms", "back", "eyes", "throat", "feet", "hands", "skin", "neck"]

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = [("assistant", "👋 Hi! I'm your medical assistant. Tell me what you're experiencing.")]
    st.session_state.symptom_info = {"symptom": "", "location": "", "duration": "", "severity": ""}
    st.session_state.step = 0

if st.button("🔄 Start Over"):
    st.session_state.messages = [("assistant", "👋 Hi! I'm your medical assistant. Tell me what you're experiencing.")]
    st.session_state.symptom_info = {"symptom": "", "location": "", "duration": "", "severity": ""}
    st.session_state.step = 0
    st.experimental_rerun()

st.title("🩺 Medical Chatbot")

# Display messages
for role, msg in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(msg)

# Department matcher
def retrieve_department(user_input):
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    similarity = util.cos_sim(user_embedding, symptom_embeddings)
    idx = int(similarity.argmax())
    return df.iloc[idx]['Department']

# Chat flow
if prompt := st.chat_input("Describe your symptoms..."):
    st.session_state.messages.append(("user", prompt))
    user_msg = prompt.lower().strip()
    info = st.session_state.symptom_info
    step = st.session_state.step

    greetings = ["hi", "hello", "hey"]
    if user_msg in greetings:
        response = "Hello! 👋 Tell me what symptoms you're experiencing."
        st.session_state.messages.append(("assistant", response))
        st.stop()

    # Step 0: Try to auto-extract symptom and location
    if step == 0:
        info["symptom"] = prompt

        found_part = next((bp for bp in body_parts if bp in user_msg), None)
        if found_part:
            info["location"] = found_part
            response = "How long has this been going on?"
            st.session_state.step += 2  # Skip step 1
        else:
            response = "Thanks. Where in your body are you feeling this?"
            st.session_state.step += 1

    elif step == 1:
        info["location"] = prompt
        response = "How long has this been going on?"
        st.session_state.step += 1

    elif step == 2:
        info["duration"] = prompt
        response = "How severe is it? (Mild / Moderate / Severe)"
        st.session_state.step += 1

    elif step == 3:
        info["severity"] = prompt
        full_input = f"{info['symptom']} {info['location']} {info['duration']} {info['severity']}"
        department = retrieve_department(full_input)
        response = f"✅ Based on what you shared, I recommend the **{department}** department."
        st.session_state.step = 0
        st.session_state.symptom_info = {"symptom": "", "location": "", "duration": "", "severity": ""}

    st.session_state.messages.append(("assistant", response))
