import json
import streamlit as st
import requests
from PIL import Image

st.title("Image Captioning App")
uploaded_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
reference_text = st.text_area(
    "Reference captions (one per line, optional)",
    placeholder="a man riding a bike\na person on a bicycle in the street",
)

# Search method selection
col1, col2 = st.columns(2)
with col1:
    search_method = st.radio("Search Method", ["Greedy", "Beam Search"], index=0)

with col2:
    if search_method == "Beam Search":
        beam_width = st.slider("Beam Width", min_value=2, max_value=10, value=3)
    else:
        beam_width = 3

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        # Move file pointer to start
        uploaded_file.seek(0)
        references = [line.strip() for line in reference_text.splitlines() if line.strip()]
        data = {
            "search_method": search_method.lower().replace(" ", ""),
            "beam_width": beam_width,
        }
        if references:
            data["references"] = json.dumps(references)
        
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            files={"file": uploaded_file.getvalue()},  # send raw bytes
            data=data,
        )
        if response.status_code == 200:
            payload = response.json()
            caption = payload["caption"]
            st.success(f"Caption: {caption}")
            if "metrics" in payload:
                st.subheader("Metrics")
                st.json(payload["metrics"])
        else:
            st.error("Error generating caption")
