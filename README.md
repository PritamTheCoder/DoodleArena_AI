# DoodleNet — AI Doodle Recognition

**DoodleNet** is a real-time doodle recognition engine powered by a custom-trained **MobileNetV3-Small** neural network. Draw a doodle, and the AI scores it against 30 known categories from the [Google Quick, Draw!](https://quickdraw.withgoogle.com/data) dataset.

![Status](https://img.shields.io/badge/status-active-brightgreen) ![Stack](https://img.shields.io/badge/stack-FastAPI_|_Streamlit_|_PyTorch-blueviolet)

> 📐 See **[ARCHITECTURE.md](ARCHITECTURE.md)** for Mermaid diagrams of the full data flow and system design.
---

## ✨ Features

| Feature | Description |
| :--- | :--- |
| **Real-time Recognition** | Draw on a canvas and get instant AI predictions |
| **Top-5 Predictions** | See ranked confidence scores for the top 5 classes |
| **Glassmorphism UI** | Premium dark-mode Streamlit dashboard |
| **REST API** | FastAPI backend with health checks, batch inference, and Swagger docs |
| **Docker Ready** | One command to build and run everything |

## 🧠 The AI Model

* **Architecture:** MobileNetV3-Small (custom-trained)
* **Input:** 96×96 Greyscale bitmaps
* **Classes:** 30 doodle categories (Cat, Dog, Pizza, House, etc.)
* **Preprocessing:** Binarize → BBox Crop → Center/Pad → Resize → Invert → Normalize

## 📂 Project Structure

```
DoodleArena_AI/
├── ai_service/
│   ├── main.py            # FastAPI app (REST API)
│   ├── app.py             # Streamlit UI (Glassmorphism dashboard)
│   ├── model.py           # MobileNetV3 architecture & loader
│   ├── preprocessor.py    # Image preprocessing pipeline
│   ├── utils.py           # Confidence calculation & top-k
│   ├── model/
│   │   └── best.pth       # Trained model weights (not in git)
│   ├── requirements.txt
│   └── Dockerfile
├── docker_compose.yml
└── README.md
```

## 🚀 Quick Start

### Prerequisites
* **Docker Desktop** installed and running
* A trained model file (`best.pth`) placed in `ai_service/model/`

### Run with Docker

```bash
git clone https://github.com/yourusername/DoodleArena_AI.git
cd DoodleArena_AI
docker-compose -f docker_compose.yml up --build -d
```

### Access

| Service | URL | Description |
| :--- | :--- | :--- |
| **Streamlit UI** | `http://localhost:8501` | Interactive drawing + recognition |
| **FastAPI Docs** | `http://localhost:8001/docs` | Swagger API documentation |
| **Health Check** | `http://localhost:8001/health` | Model status |

### Run Locally (no Docker)

```bash
cd ai_service
pip install -r requirements.txt

# Terminal 1 — API
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — UI
streamlit run app.py
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Health check |
| `GET` | `/health` | Detailed health + model info |
| `GET` | `/classes` | List all 30 classes |
| `POST` | `/recognize` | Recognize a single doodle (base64) |
| `POST` | `/batch_recognize` | Batch recognition |
| `POST` | `/visualize` | Returns the 96×96 image the model actually sees |

### Example Request

```bash
curl -X POST http://localhost:8001/recognize \
  -H "Content-Type: application/json" \
  -d '{"prompt": "cat", "image_base64": "<base64_string>"}'
```

## 📄 License

MIT 