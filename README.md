# Media Analysis Toolkit: Deepfake, AI-Generated Content, and Spam Detection


A comprehensive multimodal detection system for analyzing various media types using state-of-the-art AI models. This toolkit helps identify:
- 🖼️ Deepfake/AI-generated images
- 📝 AI-generated text
- 🎥 Deepfake videos
- 📞 Spam/Fraudulent audio

## Key Features

### 🕵️ Multi-Modal Detection
- **Image Analysis**  
  - Deepfake detection (ResNet-18 model)
  - AI-generated detection (NYUAD model)
- **Text Verification**  
  - Custom Desklib AI-text detector
- **Video Authentication**  
  - VideoMAE-based deepfake detection
- **Audio Analysis**  
  - Live call monitoring
  - Spam/fraud classification using Gemini AI

### 🔥 Advanced Capabilities
- Real-time audio processing
- Multi-model consensus for images
- Frame sampling for video analysis
- Confidence threshold customization

## Installation

1. **Clone Repository**

git clone https://github.com/mithra009/Deep-Fake-and-fraud-detection
cd media-analysis-toolkit 

2. **Install Requirements**
pip install -r requirements.txt

3. **API Keys Setup**
Create .env file:
HUGGINGFACE_TOKEN=your_hf_token
GOOGLE_API_KEY=your_google_ai_key 

4. **Usage**
Launch Application
streamlit run app.py


## Contributing
We welcome contributions! Please follow these steps:
Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit changes (git commit -m 'Add amazing feature')
Push to branch (git push origin feature/AmazingFeature)
Open a Pull Request


## Disclaimer
This toolkit provides probabilistic analysis and should not be used as sole evidence for critical decisions. Always verify important findings through multiple channels.

## Note: Requires separate API keys for Hugging Face and Google Gemini services. Model accuracy may vary based on input quality and model training data.


