import os
import uuid
import torch
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from transformers.models.vits import VitsModel
from transformers import AutoTokenizer as TtsTokenizer
import soundfile as sf


# ==============================
# Streamlit Page Setup & CSS
# ==============================
st.set_page_config(
    page_title="Smart Narration Video Processor",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark/light mode and buttons
st.markdown("""
<style>
body {font-family: 'Poppins', sans-serif;}
.main-title {font-size: 2.2rem; font-weight: 700; text-align: center; margin-bottom: 20px; color: #ffffff;}
.sub-title {font-size: 1.3rem; font-weight: 500; margin-bottom: 10px; color: #ffffff;}
.stButton>button {
    background: linear-gradient(to right, #4f46e5, #06b6d4);
    color: white; font-weight: 600; border-radius: 10px;
    padding: 0.6em 1.2em; border: none; transition: 0.3s;
}
.stButton>button:hover {transform: scale(1.05); background: linear-gradient(to right, #4338ca, #0891b2);}
.upload-widget {border: 2px dashed #ffffff; border-radius: 10px; padding: 20px;}
</style>
""", unsafe_allow_html=True)

# Sidebar for page navigation
page = st.sidebar.radio("Navigate Pages", ["Home", "About", "Contact"])

# ==============================
# Load Models (cached)
# ==============================
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO("yolov8n.pt")
    tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng").to(device)
    tts_tokenizer = TtsTokenizer.from_pretrained("facebook/mms-tts-eng")
    return device, model, tts_model, tts_tokenizer

device, model_official, tts_model, tts_tokenizer = load_models()
object_real_height = {"person": 1.7, "car": 1.5, "bus": 3.0, "dog": 0.5, "cat": 0.3, "bench": 1.0}

# ==============================
# Helper Functions
# ==============================
def get_narration(detected_objects):
    narrations = {
        "person": "There is a person ahead. Please stay alert.",
        "car": "There is a car nearby. Be cautious.",
        "bus": "There is a bus approaching. Stay safe.",
        "dog": "There is a dog nearby. Please be careful.",
        "cat": "There is a cat nearby. Watch your step."
    }
    msgs = [narrations.get(obj.lower(), f"A {obj} was detected.") for obj in detected_objects]
    return " ".join(msgs)

def tts_to_wav(text, filename):
    inputs = tts_tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = tts_model(**inputs)
    wav_array = output.waveform[0].cpu().numpy()
    wav_int16 = (wav_array * 32767).astype(np.int16)
    sf.write(filename, wav_int16, samplerate=16000)
    return filename

def draw_boxes(frame, results, focal_length=800):
    for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
        x1, y1, x2, y2 = box.astype(int)
        label = results.names[int(cls)]
        H_real = object_real_height.get(label.lower(), None)
        if H_real:
            H_img = y2 - y1
            dist = (H_real * focal_length) / H_img
            text = f"{label}: {dist:.1f}m"
            color = (0,0,255) if dist<3 else (0,255,0) if dist<7 else (255,0,0)
        else:
            text, color = label, (0,255,255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

def process_video(input_path, frame_skip=5):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w,h = int(cap.get(3)), int(cap.get(4))

    temp_video = f"{uuid.uuid4()}_video.mp4"
    out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps/frame_skip, (w,h))

    combined_audio_path = f"{uuid.uuid4()}_narration.wav"
    last_narration = ""
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_id % frame_skip == 0:
            results = model_official(frame)[0]
            labels = [results.names[int(c)] for c in results.boxes.cls.cpu().numpy().astype(int)]
            frame = draw_boxes(frame, results)
            if labels:
                narration = get_narration(labels)
                if narration != last_narration:
                    tts_to_wav(narration, combined_audio_path)
                    last_narration = narration
            for _ in range(frame_skip): out.write(frame)
        frame_id +=1

    cap.release()
    out.release()

    final_video = f"{uuid.uuid4()}_final.mp4"
    clip = VideoFileClip(temp_video)
    audio = AudioFileClip(combined_audio_path)
    if audio.duration > clip.duration: audio = audio.subclip(0, clip.duration)
    final_clip = clip.set_audio(audio)
    final_clip.write_videofile(final_video, codec="libx264", audio_codec="aac")
    clip.close(); audio.close(); final_clip.close()

    return final_video, combined_audio_path

# ==============================
# Page Logic
# ==============================
if page == "Home":
    st.markdown("<h1 style='text-align:center;'>🎥 Smart Narration Video Processor</h1>", unsafe_allow_html=True)
    st.write("Upload a video and let AI detect objects + generate audio narration.")

    file = st.file_uploader("📤 Choose a video", type=["mp4","mov","avi","mkv"])
    if file:
        st.video(file)
        if st.button("🚀 Process Video"):
            with open(file.name,"wb") as f: f.write(file.read())
            with st.spinner("Processing video... this may take a while ⏳"):
                video_out, audio_out = process_video(file.name)
            st.success("✅ Processing complete!")
            st.video(video_out)
            st.audio(audio_out)

elif page == "About":
    st.markdown("<h1 style='text-align:center;'>About This Project</h1>", unsafe_allow_html=True)
    st.write("""
**Smart Narration Video Processor** uses **YOLOv8** for object detection and **VITS TTS** for audio narration.  
It helps visualize and narrate objects in videos, useful for AI-assisted analysis or accessibility.
""")

elif page == "Contact":
    st.markdown("<h1 style='text-align:center;'>Contact Me</h1>", unsafe_allow_html=True)
    st.markdown("""
- **Name:** Borigi Jyothiradhithya  
- **Email:** adhithyaborigi@gmail.com  
- **Phone:** 7207180221
""")
    st.markdown("### Send a Message")
    with st.form(key='contact_form'):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Message")
        submitted = st.form_submit_button("Send Message")
        if submitted:
            st.success("Thank you! Your message has been sent. 📬")

