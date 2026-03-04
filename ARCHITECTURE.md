# Architecture

Technical overview of DoodleNet's data flow and system design.

## System Overview

```mermaid
graph LR
    subgraph Client
        A["🖌️ Streamlit UI<br/>(Drawing Canvas)"]
    end

    subgraph AI Service
        B["FastAPI<br/>(REST API)"]
        C["Preprocessor"]
        D["MobileNetV3-Small"]
        E["Post-Processing"]
    end

    A -- "Base64 PNG" --> B
    B --> C
    C -- "Tensor (1,1,96,96)" --> D
    D -- "Logits (1,30)" --> E
    E -- "JSON Response" --> A

    style A fill:#7c3aed,stroke:#5b21b6,color:#fff
    style B fill:#2563eb,stroke:#1d4ed8,color:#fff
    style C fill:#0891b2,stroke:#0e7490,color:#fff
    style D fill:#059669,stroke:#047857,color:#fff
    style E fill:#d97706,stroke:#b45309,color:#fff
```

## Preprocessing Pipeline

The preprocessing exactly mirrors the training pipeline to avoid train/inference skew.

```mermaid
graph TD
    A["Raw Canvas Image<br/>(448×448 RGBA)"] --> B["RGB Conversion<br/>(RGBA → white-bg RGB)"]
    B --> C["Grayscale<br/>(RGB → L)"]
    C --> D["Binarize<br/>(threshold = 200)"]
    D --> E["BBox Crop<br/>(tight ink region)"]
    E --> F["Center + Pad<br/>(square, white bg)"]
    F --> G["Resize<br/>(→ 96×96, INTER_AREA)"]
    G --> H["Normalize<br/>(float → invert → [-1,1])"]
    H --> I["Tensor<br/>(1, 96, 96)"]

    style A fill:#7c3aed,stroke:#5b21b6,color:#fff
    style I fill:#059669,stroke:#047857,color:#fff
```

## Model Architecture

| Property | Value |
| :--- | :--- |
| **Base Model** | MobileNetV3-Small (torchvision) |
| **Input** | `(B, 1, 96, 96)` — single-channel grayscale |
| **Output** | `(B, 30)` — logits over 30 doodle classes |
| **First Conv** | Modified: 3 → 1 input channels |
| **Classifier** | Standard MobileNetV3 head → 30 classes |
| **Parameters** | ~1.5M total |
| **Inference** | CPU: ~5ms, GPU: ~1ms |

### Modifications from Standard MobileNetV3

```python
# 1. Single-channel input (grayscale doodles, not RGB photos)
model.features[0][0] = Conv2d(1, 16, kernel_size=3, stride=2, padding=1)

# 2. 30-class output (Quick, Draw! categories, not ImageNet 1000)
model.classifier[-1] = Linear(1024, 30)
```

## API Endpoints

```mermaid
graph LR
    subgraph "GET"
        A["/ (health)"]
        B["/health (detailed)"]
        C["/classes"]
    end

    subgraph "POST"
        D["/recognize"]
        E["/batch_recognize"]
        F["/visualize"]
    end

    style A fill:#059669,stroke:#047857,color:#fff
    style B fill:#059669,stroke:#047857,color:#fff
    style C fill:#059669,stroke:#047857,color:#fff
    style D fill:#2563eb,stroke:#1d4ed8,color:#fff
    style E fill:#2563eb,stroke:#1d4ed8,color:#fff
    style F fill:#7c3aed,stroke:#5b21b6,color:#fff
```

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/` | GET | Quick health check |
| `/health` | GET | Model status + metadata |
| `/classes` | GET | List all 30 recognizable classes |
| `/recognize` | POST | Single doodle → confidence + top-5 |
| `/batch_recognize` | POST | Multiple doodles in one request |
| `/visualize` | POST | Returns the 96×96 preprocessed image the model sees |

## Doodle Classes (30)

```
Animals:   cat, dog, bird, fish, cow
Food:      apple, banana, pizza, cake, ice cream
Vehicles:  car, bicycle, airplane, bus, train
Scenery:   house, tree, flower, sun, cloud
Shapes:    star, moon, hand, face, clock
Objects:   book, chair, shoe, key, umbrella
```

## Project Layout

```
DoodleArena_AI/
├── ai_service/
│   ├── main.py            # FastAPI application + endpoints
│   ├── app.py             # Streamlit UI (Glassmorphism)
│   ├── model.py           # MobileNetV3Doodle architecture
│   ├── preprocessor.py    # Image → Tensor pipeline
│   ├── utils.py           # Softmax, top-k, class lookups
│   ├── model/best.pth     # Trained weights (gitignored)
│   ├── Dockerfile
│   └── requirements.txt
├── docker_compose.yml
├── ARCHITECTURE.md         # ← You are here
└── README.md
```
