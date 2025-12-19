# 🎥 UTube Assistant

A powerful AI-powered assistant that lets you **"chat" with YouTube videos**.  
It ingests video transcripts, understands the context using **RAG (Retrieval-Augmented Generation)**, and can even generate images based on the video's visual style using a **local Stable Diffusion server**.

![Overview](https://github.com/zuhthisahan/Utube-assistant/blob/sahan/generated_images/ui.png)

---

## 📺 Demo

👉 **Demo Video:** 
![demo](https://github.com/zuhthisahan/Utube-assistant/blob/sahan/demo.gif)


---

## 🚀 Features

- **Video Ingestion** – Paste a YouTube URL and index the transcript into a vector database
- **Contextual Q&A** – Ask questions about timestamps, summaries, or details
- **Visual Generation** – Generate images using Stable Diffusion (AUTOMATIC1111)
- **Data Analysis Tools** – Plot word-frequency trends from transcripts
- **External Knowledge** – Wikipedia enrichment when video context is insufficient
- **Hybrid Architecture** – Docker + Host GPU for heavy image generation

---

## 🛠️ Tech Stack

- **Frontend:** React (Vite), Tailwind CSS, Lucide React  
- **Backend:** Python 3.13, FastAPI, Uvicorn  
- **AI Orchestration:** LangChain, LangGraph  
- **Vector DB:** ChromaDB (Local)  
- **Image Generation:** Stable Diffusion WebUI (AUTOMATIC1111)  
- **LLM:** LM Studio (Local) or OpenAI  
- **Infrastructure:** Docker & Docker Compose

---

## ⚙️ Prerequisites

1.  **Git:** To clone the repo.
2.  **Docker Desktop:** For the containerized setup.
3.  **Automatic1111 (Stable Diffusion):**
    * **Crucial:** Must be running with the `--api` flag.
    * Edit your `webui-user.bat` and set: `set COMMANDLINE_ARGS=--api`
    * Set up the correct python version (3.10.x) at `webui-user.bat'
4.  **LM Studio (Or any LLM):**
    * Start the local server on port `1234` or adjust accordingly.

---

## 🚀 Setup Option 1: Docker (Recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/zuhthisahan/Utube-assistant.git
cd Utube-assistant/video-assistant
```

---

### 2. Configure Environment Variables

Create a `.env` file in the `video-assistant` folder (same level as `docker-compose.yml`).

```env
# OpenAI (Optional) If you use
OPENAI_API_KEY=sk-proj-your-key-here

# Docker → Host
SD_BASE_URL=http://host.docker.internal:7860
LM_STUDIO_URL=http://host.docker.internal:1234/v1
```

---

### 3. Start Required AI Servers (Host Machine)

**Stable Diffusion**  
- Run `webui-user.bat`
- Confirm: http://127.0.0.1:7860

**LM Studio (Optional)**  
- Start Local Inference Server on port `1234`

---

### 4. Run the Application

```bash
docker-compose up --build
```

- Frontend: http://localhost:3000  
- Backend Docs: http://localhost:8000/docs

---

## 💻 Setup Option 2: Manual (Development)

### Backend

```bash
cd backend
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

Run server:
From the main project folder
```bash
python -m backend.main  
```

---

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: http://localhost:5173

---

### Manual Mode Configuration Note

When not using Docker, `host.docker.internal` does not exist.

```python
# tools.py
base_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
```

---

## 🎨 Stable Diffusion API Check

Open in browser:
```
http://127.0.0.1:7860/docs
```

If Swagger UI appears, the API is working ✅

---

## ✅ Summary

- Docker setup = easiest & recommended
- Manual setup = best for development
- Stable Diffusion runs on host GPU
- LM Studio is optional but powerful


