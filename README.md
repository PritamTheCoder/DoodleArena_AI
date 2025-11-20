# DoodleArena AI

**DoodleArena AI** is a real-time, 2-player competitive drawing game powered by Deep Learning. 
Players are given a prompt (e.g., "Draw a Cat"), and a MobileNetV3 neural network scores their drawings in real-time. The player with the best drawing accuracy wins the round.

![Project Status](https://img.shields.io/badge/status-development-orange) ![Tech Stack](https://img.shields.io/badge/stack-Laravel_|_FastAPI_|_AlpineJS-blue)

## System Architecture

The application is split into two distinct services:

1.  **Game Engine (Laravel 10+)**: Handles authentication, game state, lobbies, and WebSocket broadcasting (using Laravel Reverb or Pusher).
2.  **AI Vision Service (FastAPI)**: A stateless microservice that accepts a Base64 image and returns a confidence score.


## Quick Start (Local Hosting)

To host this for friends on your local network:

### Prerequisites
* Docker Desktop installed and running.
* Git.

### Installation

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/yourusername/DoodleArena_AI.git](https://github.com/yourusername/DoodleArena_AI.git)
    cd DoodleArena_AI
    ```

2.  **Add the Model:**
    Place your trained PyTorch model (`best.pth`) inside `ai_service/models/`.

3.  **Start the Services:**
    ```bash
    docker-compose up --build -d
    ```

4.  **Access the Game:**
    * **Host (You):** Go to `http://localhost:8000`
    * **Friends:** Find your local IP (e.g., `ipconfig` on Windows -> IPv4 Address, usually `192.168.x.x`).
    * Friends should visit: `http://192.168.x.x:8000`

## Project Structure

| Service | Path | Port | Description |
| :--- | :--- | :--- | :--- |
| **Web App** | `/web_app` | `8000` | Laravel Backend & Frontend (Blade + Alpine.js) |
| **AI API** | `/ai_service` | `8001` | Python FastAPI (Torch Inference) |
| **Database** | N/A | `3306` | MySQL 8.0 |
| **Redis** | N/A | `6379` | Queue & WebSocket Broadcasting |

## The AI Model

* **Architecture:** MobileNetV3 (Small) custom-trained on the Google QuickDraw dataset.
* **Input:** 96x96 Greyscale bitmaps.
* **Classes:** 30 common doodle categories (Cat, House, Sun, etc.).
* **Preprocessing:** The pipeline simulates the 28x28 pixelation of the original dataset to ensure high accuracy even with rough mouse drawings.

## How to Play

1.  Create an account or Login.
2.  Click **"Create Room"** to generate a unique Room Code.
3.  Share the code with a friend.
4.  Once Player 2 joins, the game begins!
5.  **Goal:** Reach 2 points first.
    * Round 1: "Draw a Pizza" -> Scores compared -> Point awarded.
    * Round 2: "Draw a Car" -> ...