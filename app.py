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

st.title("🩺 AI Medical Chatbot")
st.markdown("Describe your symptoms, and I'll guide you to the right department.")

with st.form("symptom_form"):
    symptom = st.text_area("What symptom(s) are you experiencing?")
    location = st.text_input("Where is the symptom located?")
    duration = st.selectbox("How long has it been happening?", ["< 1 day", "1-3 days", "1 week", "More than a week"])
    severity = st.radio("How severe is it?", ["Mild", "Moderate", "Severe"])
    submitted = st.form_submit_button("Get Recommendation")

def retrieve_department(user_input):
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    similarity = util.cos_sim(user_embedding, symptom_embeddings)
    idx = int(similarity.argmax())  
    return df.iloc[idx]['Department']


if submitted:
    full_input = f"{symptom} {location} {duration} {severity}"
    dept = retrieve_department(full_input)
    st.success(f"📌 You should consult the **{dept}** department.")
