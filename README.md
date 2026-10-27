# AI Image Authenticity Detector

[🌐 👉 CLICK HERE FOR LIVE DEMO 👈]( https://gobalakrishnan-p.github.io/AI-Image-Detector/)

A clean, privacy-first web application that analyzes uploaded images in real-time to predict whether an image is **Likely AI-Generated / Manipulated**, **Likely Authentic**, or **Uncertain**.

---

## 📌 Project Overview

- **Project Name**: AI Image Authenticity Detector
- **Target Architecture**: 100% Client-Side In-Browser Neural Inference (Privacy-First)
- **Technology Stack**: HTML5, CSS3, Vanilla JavaScript (ES Modules), [Transformers.js](https://huggingface.co/docs/transformers.js), and ONNX Runtime Web (WASM).
- **No Backend Server Required**: No Python, Node.js, React, Firebase, or external paid APIs.

---

## 🚀 Key Features

1. **Privacy-First In-Browser Processing**:
   - The user selects an image from their device.
   - The image is processed locally in the browser memory using WebAssembly & ONNX Runtime.
   - Images are **never** uploaded to any external backend or database.

2. **AI Model & Classification**:
   - Utilizes `onnx-community/CommunityForensics-DeepfakeDet-ViT-ONNX` via the Transformers.js `image-classification` pipeline.
   - Evaluates real vs. synthetic class scores and inspects output labels carefully.
   - Handles ambiguity: displays an **"Uncertain"** status if confidence is below 60% or when predictions are close, avoiding false certainties.

3. **User Experience & Design**:
   - Professional dark cybersecurity research aesthetic.
   - Drag-and-drop file upload with format validation (JPG, JPEG, PNG, WEBP) and size limits (15MB).
   - Real-time image preview with metadata details (dimensions, file size, file name).
   - Non-blocking asynchronous loading indicator with model progress updates.
   - Visual confidence progress bar and an expandable **Technical Analysis** details accordion.
   - Responsible AI disclaimer noting that detections are probabilistic predictions.

---

## 📂 Project Structure

```
/
├── index.html     # Semantic HTML5 layout and structure
├── style.css      # Cybersecurity dark theme & responsive mobile styling
├── script.js      # Transformers.js pipeline, image validation & result logic
└── README.md      # Documentation and usage guide
```

---

## 🛠️ How to Run the Project

Since this is a modern vanilla web application using ES Modules and WebAssembly:

### Option 1: Using Python Built-in HTTP Server
```bash
python -m http.server 8000
```
Then open `http://localhost:8000` in your web browser.

### Option 2: Using VS Code Live Server
1. Open the project folder in VS Code.
2. Right-click `index.html` and click **"Open with Live Server"**.

---

## 🛡️ Responsible AI Note
Deepfake and synthetic image detection algorithms are probabilistic and may produce false positives or false negatives. Predictions should be considered supplementary forensic indicators rather than definitive proof.
