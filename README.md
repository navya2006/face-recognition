# 🔐 FaceLock - Face Recognition System

A Flask-based real-time face recognition web application that allows users to register their faces and identify previously registered users using OpenCV's Haar Cascade face detector and LBPH Face Recognizer.

## ✨ Features

- 📷 Live webcam feed
- 👤 Register new users with face capture
- 🧠 Train an LBPH face recognition model
- 🔍 Recognize registered users in real time
- 🎨 Modern cyberpunk-inspired web interface
- ⚡ Built with Flask and OpenCV

---

## 🛠️ Tech Stack

- Python
- Flask
- OpenCV
- OpenCV Contrib (LBPH Face Recognizer)
- NumPy
- HTML/CSS/JavaScript

---

## 📁 Project Structure

```
face-recognition/
│── app.py
│── face_model.py
│── requirements.txt
│── .gitignore
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
├── dataset/          # Generated face images (ignored)
├── trainer.yml       # Trained LBPH model (ignored)
└── labels.npy        # User labels (ignored)
```

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/navya2006/face-recognition.git
cd face-recognition
```

### Create a virtual environment

```bash
python3 -m venv venv
```

Activate it:

**macOS/Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 📸 How to Use

### Register a Face

1. Enter your **Name**
2. Enter a unique **User ID**
3. Click **Capture & Train**
4. The application captures your face and trains the recognition model.

### Recognize a Face

1. Stand in front of the camera.
2. Click **Identify Face**.
3. The system predicts the registered person's identity.

---

## 📦 Requirements

- Python 3.10+ (recommended)
- Webcam
- OpenCV Contrib

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## ⚠️ Notes

- The `dataset/`, `trainer.yml`, and `labels.npy` files are generated after registering users and are excluded from version control.
- If the webcam does not appear, ensure camera permissions are enabled for your terminal or IDE.
- If you encounter issues with Haar Cascade files, use a stable version of `opencv-contrib-python` (e.g. `4.10.0.84`).

---