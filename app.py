import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import re

# Load model and CSV
@st.cache_resource
def load_model_and_data():
    df = pd.read_csv("medical_symptoms_updated.csv")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["Symptom Description"].tolist(), convert_to_tensor=True)
    return df, model, embeddings

df, model, symptom_embeddings = load_model_and_data()

# Body parts and pain types for simple keyword detection
body_parts = df["Location"].unique().tolist()
pain_types = df["Pain Type"].unique().tolist()

def get_department(query):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarity = util.cos_sim(query_embedding, symptom_embeddings)
    best_match = int(similarity.argmax())
    return df.iloc[best_match]["Department"]

# Chat state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.step = -1
    st.session_state.data = {"symptom": "", "location": "", "duration": "", "pain_type": ""}
    st.session_state.pending_user_input = None

st.title("🩺 AI Medical Chatbot")

if st.button("🔄 Restart Chat"):
    st.session_state.messages = []
    st.session_state.step = -1
    st.session_state.data = {"symptom": "", "location": "", "duration": "", "pain_type": ""}
    st.session_state.pending_user_input = None
    st.experimental_rerun()

# Show chat
for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(text)

# Capture user input only once per cycle
user_input = st.chat_input("Type your message...")
if user_input and st.session_state.pending_user_input is None:
    st.session_state.pending_user_input = user_input

if st.session_state.pending_user_input:
    user_input = st.session_state.pending_user_input
    st.session_state.messages.append(("user", user_input))
    user_input_lower = user_input.lower().strip()
    step = st.session_state.step
    data = st.session_state.data

    # STEP -1: Waiting for first user message
    if step == -1:
        data["symptom"] = user_input
        found_location = next((bp for bp in body_parts if bp in user_input_lower), None)
        if found_location:
            data["location"] = found_location
            st.session_state.step = 2
            st.session_state.messages.append(("assistant", "How long have you been experiencing this?"))
        else:
            st.session_state.step = 1
            st.session_state.messages.append(("assistant", "Where in your body is this occurring?"))

    # STEP 1: LOCATION
    elif step == 1:
        data["location"] = user_input
        st.session_state.step = 2
        st.session_state.messages.append(("assistant", "How long have you been experiencing this?"))

    # STEP 2: DURATION
    elif step == 2:
        if re.search(r"\b(\d+\s*(day|week|hour|month|minute|year)s?|today|yesterday|this morning|few hours|couple of days)\b", user_input_lower):
            data["duration"] = user_input
            st.session_state.step = 3
            st.session_state.messages.append(("assistant", "What type of pain is it? (e.g., sharp, dull, throbbing, burning, etc.)"))
        else:
            st.session_state.messages.append(("assistant", "Please enter a realistic duration (e.g., '3 days', 'since yesterday', '1 hour')."))

    # STEP 3: PAIN TYPE
    elif step == 3:
        found_pain = next((pt for pt in pain_types if pt in user_input_lower), None)
        if found_pain:
            data["pain_type"] = found_pain
        else:
            data["pain_type"] = user_input  # Accept as-is

        # Final prediction
        full_query = f"{data['symptom']} {data['location']} {data['duration']} {data['pain_type']}"
        dept = get_department(full_query)
        st.session_state.messages.append(("assistant", f"✅ Based on what you've told me, I recommend visiting the **{dept}** department."))

        # Reset for new conversation
        st.session_state.step = -1
        st.session_state.data = {"symptom": "", "location": "", "duration": "", "pain_type": ""}

    # Clear pending input
    st.session_state.pending_user_input = None
