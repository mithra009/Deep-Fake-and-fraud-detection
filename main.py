
import streamlit as st
from PIL import Image
import warnings
import pyaudio
import wave
import torch
import numpy as np
from torchvision import transforms
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, pipeline, AutoTokenizer, AutoConfig, AutoModel
from transformers import AutoProcessor, AutoModelForVideoClassification
import google.generativeai as genai
from time import sleep
import torch.nn as nn
from transformers import PreTrainedModel
import cv2
import os

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ====== Model Setup ======
HUGGINGFACE_TOKEN = ""  # Replace with your token
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper function to handle model loading
def load_model(model_class, model_path, device, is_pipeline=False, **kwargs):
    if is_pipeline:
        return pipeline(**kwargs)
    
    try:
        model = model_class.from_pretrained(model_path, **kwargs)
        # Check if model has meta tensors and handle appropriately
        if hasattr(model, "_is_meta") and model._is_meta:
            model = model.to_empty(device=device)
        else:
            model = model.to(device)
        return model
    except NotImplementedError as e:
        if "Cannot copy out of meta tensor" in str(e):
            model = model_class.from_pretrained(model_path, **kwargs)
            model = model.to_empty(device=device)
            return model
        raise e

# Deepfake Image Detection (ResNet-18)
model_id_deepfake = "dima806/deepfake_vs_real_image_detection"
feature_extractor_deepfake = AutoFeatureExtractor.from_pretrained(model_id_deepfake, token=HUGGINGFACE_TOKEN)
model_deepfake = load_model(
    AutoModelForImageClassification, 
    model_id_deepfake, 
    device, 
    token=HUGGINGFACE_TOKEN
)

# AI-Generated Image Detection (NYUAD)
model_id_nyuad = "NYUAD-ComNets/NYUAD_AI-generated_images_detector"
classifier_nyuad = load_model(
    None,
    None,
    device,
    is_pipeline=True,
    task="image-classification", 
    model=model_id_nyuad
)

# AI-Generated Text Detection (Desklib)
class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def _init_(self, config):
        super()._init_(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())
        return {"logits": logits, "loss": loss}

model_directory_desklib = "desklib/ai-text-detector-v1.01"
tokenizer_desklib = AutoTokenizer.from_pretrained(model_directory_desklib)
model_desklib = load_model(
    DesklibAIDetectionModel, 
    model_directory_desklib, 
    device
)

# Deepfake Video Detection (VideoMAE)
model_id_videomae = "Ammar2k/videomae-base-finetuned-deepfake-subset"
processor_videomae = AutoProcessor.from_pretrained(model_id_videomae, token=HUGGINGFACE_TOKEN)
model_videomae = load_model(
    AutoModelForVideoClassification, 
    model_id_videomae, 
    device, 
    token=HUGGINGFACE_TOKEN
)

# Google Gemini Setup
genai.configure(api_key="") 
model_gemini = genai.GenerativeModel("gemini-2.0-flash")

# ====== Streamlit Config ======
st.set_page_config(page_title="Media Analysis App", page_icon=":movie_camera:", layout="centered")

# ====== Sidebar Navigation ======
st.sidebar.title("Choose Analysis Type")
feature = st.sidebar.selectbox("Select a feature", [
    "Deepfake & AI-Generated Image Detection",
    "AI-Generated Text Detection",
    "Deepfake Video Detection",
    "Live Audio Analysis",
    "Spam Audio Detection"
])

# ====== Deepfake Image Detection Logic (ResNet-18) ======
def classify_image_deepfake(image, fake_bias=10.0, confidence_threshold=75):
    image = image.convert("RGB")
    inputs = feature_extractor_deepfake(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the same device as model
    with torch.no_grad():
        outputs = model_deepfake(**inputs)
    logits = outputs.logits
    logits[0][1] += fake_bias
    probs = torch.softmax(logits, dim=1)[0]
    predicted_id = torch.argmax(probs).item()
    confidence = probs[predicted_id].item() * 100
    if confidence >= confidence_threshold:
        label = model_deepfake.config.id2label[predicted_id]
    else:
        label = "undetermined"
    return label, confidence

# ====== AI-Generated Image Detection Logic (NYUAD) ======
def classify_image_nyuad(image):
    try:
        pred = classifier_nyuad(image)
        top_pred = max(pred, key=lambda x: x['score'])
        label = top_pred['label']
        confidence = top_pred['score'] * 100
        return label, confidence
    except Exception as e:
        return "error", str(e)

# ====== Combined Image Detection ======
if feature == "Deepfake & AI-Generated Image Detection":
    st.title("Deepfake & AI-Generated Image Detection")
    st.info("Upload an image to detect whether it's a deepfake or AI-generated")

    uploaded_file = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Deepfake Detection
        try:
            deepfake_label, deepfake_confidence = classify_image_deepfake(image, fake_bias=1.0, confidence_threshold=75)
            if deepfake_label == "undetermined":
                st.warning(f"⚠ Deepfake Model: UNDETERMINED ({deepfake_confidence:.2f}% confidence)")
            else:
                st.success(f"Deepfake Model: {deepfake_label.upper()} ({deepfake_confidence:.2f}% confidence)")
        except Exception as e:
            st.error(f"Error in deepfake detection: {e}")

        # AI-Generated Detection
        try:
            nyuad_label, nyuad_confidence = classify_image_nyuad(image)
            if nyuad_label == "error":
                st.error(f"AI-Generated Model Error: {nyuad_confidence}")
            else:
                st.success(f"AI-Generated Model: {nyuad_label.upper()} ({nyuad_confidence:.2f}% confidence)")
        except Exception as e:
            st.error(f"Error in AI-generated detection: {e}")

# ====== AI-Generated Text Detection Logic ======
def predict_single_text(text, model, tokenizer, device, max_len=768, threshold=0.5):
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        probability = torch.sigmoid(logits).item()
    label = "AI Generated" if probability >= threshold else "Not AI Generated"
    return probability, label

if feature == "AI-Generated Text Detection":
    st.title("AI-Generated Text Detection")
    st.info("Enter text to detect whether it is AI-generated or human-written")

    input_text = st.text_area("Enter text for analysis", height=200)
    if st.button("Analyze Text"):
        if input_text:
            try:
                probability, label = predict_single_text(input_text, model_desklib, tokenizer_desklib, device)
                st.success(f"Prediction: {label} (Probability of being AI-generated: {probability:.4f})")
            except Exception as e:
                st.error(f"Error in text detection: {e}")
        else:
            st.warning("Please enter some text to analyze.")

# ====== Deepfake Video Detection Logic ======
def sample_frames_uniformly(video_path, max_frames=16):
    """
    Samples frames uniformly across the video.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < max_frames:
        max_frames = total_frames
    indices = torch.linspace(0, total_frames - 1, steps=max_frames).long().tolist()
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    cap.release()
    return frames

def classify_video(video_path, max_frames=16):
    """
    Classifies a video as real or deepfake using VideoMAE.
    """
    try:
        frames = sample_frames_uniformly(video_path, max_frames=max_frames)
        if not frames:
            return "error", "No frames could be extracted from video."
        inputs = processor_videomae(frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}  # Move inputs to the same device as model
        with torch.no_grad():
            outputs = model_videomae(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]
        predicted_id = torch.argmax(probs).item()
        confidence = probs[predicted_id].item() * 100
        label = model_videomae.config.id2label[predicted_id]
        return label, round(confidence, 2)
    except Exception as e:
        return "error", str(e)

if feature == "Deepfake Video Detection":
    st.title("Deepfake Video Detection")
    st.info("Upload a video to detect whether it's a deepfake or real")

    uploaded_file = st.file_uploader("Upload a video (MP4, AVI)", type=["mp4", "avi"])
    if uploaded_file:
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            label, confidence = classify_video(video_path)
            if label == "error":
                st.error(f"Error in video detection: {confidence}")
            else:
                st.success(f"Prediction: {label.upper()} ({confidence:.2f}% confidence)")
            os.remove(video_path)
        except Exception as e:
            st.error(f"Error processing video: {e}")

# ====== Audio Recorder ======
def record_audio(output_path, record_seconds=5, rate=44100, chunk=1024, channels=1):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    frames = [stream.read(chunk) for _ in range(int(rate / chunk * record_seconds))]
    stream.stop_stream()
    stream.close()
    p.terminate()
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))

# ====== Gemini Audio Processor ======
def process_audio(audio_path, prompt):
    try:
        audio_file = genai.upload_file(path=audio_path)
        response = model_gemini.generate_content([prompt, audio_file])
        return response.text.strip().lower()
    except Exception as e:
        return f"error: {e}"

# ====== Live Audio Analysis ======
if feature == "Live Audio Analysis":
    st.title("Live Audio Analysis with Generative AI")
    st.write("🎙 Record audio to classify it as 'Spam', 'Fraud', or 'Genuine'.")

    audio_path = "temp_audio.wav"
    complete_audio_path = "complete_audio.wav"
    all_frames = []
    accumulated_seconds = 0

    if st.button("Start Live Analysis"):
        st.write("🔴 Recording in progress... Speak now!")
        status_placeholder = st.empty()
        result_placeholder = st.empty()

        while True:
            record_audio(audio_path, record_seconds=5)
            with wave.open(audio_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                all_frames.append(frames)
            with wave.open(complete_audio_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                wf.writeframes(b"".join(all_frames))

            prompt = 'Classify the audio as "spam", "fraud", or "genuine". Respond with one word in quotes and a short reason.'
            result = process_audio(complete_audio_path, prompt)
            accumulated_seconds += 5
            status_placeholder.text(f"Total recording time: {accumulated_seconds} seconds")

            if result:
                if "error" in result:
                    result_placeholder.error(result)
                else:
                    first_word = result.split('"')[1] if '"' in result else "unknown"
                    result_placeholder.markdown(f"### Analysis Result: {result}")
                    if first_word in ["spam", "fraud"]:
                        st.error(f"🚨 DETECTED: {first_word.upper()} — Please disconnect the call immediately!")
                        break
            else:
                result_placeholder.error("Error processing the audio. Please try again.")
            sleep(1)
        st.success("✅ Analysis complete. Recording stopped due to detection.")

# ====== Audio Upload Analysis ======
elif feature == "Spam Audio Detection":
    st.title("Audio File Classification")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    if uploaded_file:
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        prompt = 'Analyze the call and classify as "fraud", "spam", or "genuine". Respond with one word and short reason.'
        result = process_audio("temp_audio.wav", prompt)
        if "error" in result:
            st.error(result)
        else:
            st.success("Analysis Result:")
            st.write(result)
    else:
        st.write("Please upload an audio file for analysis.") 
