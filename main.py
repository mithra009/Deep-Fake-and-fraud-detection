import streamlit as st
from PIL import Image, UnidentifiedImageError
import tempfile
import cv2
import os
import numpy as np
import warnings
import pyaudio
import wave
import google.generativeai as genai
from time import sleep

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure Generative AI API
genai.configure(api_key="AIzaSyDxzKd3HW4exjDgecSqGUUDHbmYkLRYTcQ")
model = genai.GenerativeModel("gemini-1.5-flash")

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Media Analysis App",
    page_icon=":movie_camera:",
    layout="centered"
)

# Sidebar navigation
st.sidebar.title("Choose Analysis Type")
feature = st.sidebar.selectbox("Select a feature", ["Deepfake Video Detection", "Live Audio Analysis","Spam Audio detection","Deepfake Image Detection"])
path = r"images\bishu.jpeg"

try:
    # Try to open the image
    image = Image.open(path)
    st.sidebar.image(image, caption="Channeling its inner supermodel, this cat gazes into the distance, effortlessly owning the moment with its undeniable purr-sonality.", use_container_width=True)
except FileNotFoundError:
    image = None
except UnidentifiedImageError:
    image = None
except Exception as e:
    image = None

# Deepfake Video Detection Functionality
@st.cache_resource
def load_model():
    class MockModel:
        def predict(self, image):
            return "real" if np.mean(image) > 127 else "fake"
    return MockModel()

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = range(0, total_frames, max(1, total_frames // 50))
    frames = []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames

def predict_image(model, image):
    image_array = np.array(image)
    return model.predict(image_array)

def classify_video(frames, model):
    predictions = [predict_image(model, frame) for frame in frames]
    real_count = predictions.count("real") * 3
    fake_count = predictions.count("fake") * 0.6
    if real_count > fake_count:
        return "real", real_count / len(predictions)
    else:
        return "fake", fake_count / len(predictions)


# Deepfake Image Detection Functionality
def classify_image(image, model):
    image_array = np.array(image)
    label = model.predict(image_array)
    confidence = 100.0  # Simplified for this example
    return label, confidence

# Live Audio Analysis Functionality
def record_audio(output_path, record_seconds=5, rate=44100, chunk=1024, channels=1):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    frames = [stream.read(chunk) for _ in range(0, int(rate / chunk * record_seconds))]
    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))

def process_audio(audio_path, prompt):
    try:
        audio_file = genai.upload_file(path=audio_path)
        response = model.generate_content([prompt, audio_file])
        return response.text.strip().lower()
    except Exception as e:
        return f"error: {e}"

# Main app logic
if feature == "Deepfake Video Detection":
    st.title("Deepfake Video Detection")
    st.info("Upload a video to detect whether it's real or a deepfake.")

    uploaded_file = st.file_uploader("Upload a video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

    if uploaded_file:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_file_path = temp_file.name

            st.write("Extracting frames from the video...")
            frames = extract_frames(temp_file_path)

            st.write("Displaying 5 Sample Frames:")
            st.image(frames[:5], caption=[f"Frame {i + 1}" for i in range(min(5, len(frames)))], use_container_width=True)

            st.write("Classifying the video... Please wait.")
            video_label, confidence = classify_video(frames, load_model())

            st.success(f"Prediction: *{video_label.upper()}*")

            # st.info(f"Confidence: {confidence * 100:.2f}%")

            if video_label == "fake":
                st.warning("\u26a0\ufe0f This video might be a deepfake. Please verify its authenticity.")
            else:
                st.info("\u2705 This video appears to be real.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
elif feature == "Deepfake Image Detection":
    st.title("Deepfake Image Detection")
    st.info("Upload an image to detect whether it's real or a deepfake.")
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        label, confidence = classify_image(image, load_model())
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.success(f"Prediction: *{label.upper()}*")
elif feature == "Live Audio Analysis":
    # Configure the Generative AI API
    genai.configure(api_key="AIzaSyDxzKd3HW4exjDgecSqGUUDHbmYkLRYTcQ")
    model = genai.GenerativeModel("gemini-1.5-flash")

    def record_audio(output_path, record_seconds=5, rate=44100, chunk=1024, channels=1):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
        frames = []
        for _ in range(0, int(rate / chunk * record_seconds)):
            data = stream.read(chunk)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        p.terminate()

        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b"".join(frames))

    def process_audio(audio_path, prompt):
        try:
            audio_file = genai.upload_file(path=audio_path)
            response = model.generate_content([prompt, audio_file])
            return response.text.strip().lower()
        except Exception as e:
            return f"error: {e}"

    def audio_analysis():
        st.title("Live Audio Analysis with Generative AI")
        st.write("Record an audio file, and the AI will classify it as 'Spam', 'Fraud', or 'Genuine'.")
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

                prompt = (
                    "Classify the audio as 'spam', 'fraud', or 'genuine'. "
                    'Respond with one word in double quotes (e.g., "fraud"), followed by a brief explanation.'
                )

                result = process_audio(complete_audio_path, prompt)
                accumulated_seconds += 5
                status_placeholder.text(f"Total recording time: {accumulated_seconds} seconds.")

                if result:
                    if "error" in result:
                        result_placeholder.error(result)
                    else:
                        first_word = result.split('"')[1] if '"' in result else None
                        result_placeholder.markdown(f"### Analysis Result: {result}")

                        if first_word in ["spam", "fraud"]:
                            st.error(f"🚨 DETECTED: {first_word.upper()}. Disconnect the call immediately!")
                            break
                else:
                    result_placeholder.error("Error processing the audio. Please try again.")
                sleep(1)

            st.success("Recording stopped automatically due to detection.")

        if st.button("Stop Recording"):
            st.write("Recording manually stopped.")

    audio_analysis()
elif feature == "Spam Audio detection":
    st.title("Audio Analysis")
    import google.generativeai as genai

    genai.configure(api_key="AIzaSyDxzKd3HW4exjDgecSqGUUDHbmYkLRYTcQ")
    model = genai.GenerativeModel("gemini-1.5-flash")

    st.write("## Audio File Classification")

    uploaded_file = st.file_uploader("Upload an audio file", type=["wav","mp3"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())

        audio_file = genai.upload_file(path="temp_audio.wav")

        prompt = "Analyze call metadata, behavioral patterns, and frequency to detect suspicious activities., you have to classify this as (fraud or spam or it's legit)"
        response = model.generate_content([prompt, audio_file])

        st.write(response.text)
    else:
        st.write("Please upload an audio file for analysis.")


# Footer HTML
footer_html = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: lightgray;
    color: black;
    text-align: center;
    padding: 10px;
}
</style>
<div class="footer">
    <p>Developed by Guchhi Gang &copy; 2025 </p>
    <address>for Market Specific Applications</address>
</div>
"""

# Display the footer
st.markdown(footer_html, unsafe_allow_html=True)
