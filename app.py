import streamlit as st
from model_helper import predict

st.set_page_config(page_title="Vehicle Damage Detection", layout="wide")
st.title("Vehicle Damage Detection")

uploaded_file = st.file_uploader("Upload a vehicle image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Save to temp file
    temp_file = "temp_upload.jpg"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Get prediction with loading indicator
    with st.spinner("Analyzing damage..."):
        try:
            prediction = predict(temp_file)
            st.success(f"Prediction: {prediction}")
        except Exception as e:
            st.error(f"Error: {str(e)}")