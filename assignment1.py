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

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        # Move file pointer to start
        uploaded_file.seek(0)
        references = [line.strip() for line in reference_text.splitlines() if line.strip()]
        data = {}
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
