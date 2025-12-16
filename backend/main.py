import os
import re
import shutil
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage
from .tools import consult_mcp_knowledge
import json
from fastapi.responses import StreamingResponse

# Local Import. graph and state
from .graph import app as graph_app
from .state import AppState, STORE

# CONFIG 
IMG_DIR = "generated_images"
os.makedirs(IMG_DIR, exist_ok=True)

# Initialize FastAPI
app = FastAPI(title="Video Assistant Backend")

from contextlib import asynccontextmanager  # the async function we kept around

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # prepare the mcp server
#     try:
#         await consult_mcp_knowledge("warmup")
#     except Exception:
#         pass

#     yield

# 1. CORS Setup for frontend to allow request
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Allows the frontend to access http://localhost:8000/images/x.png
app.mount("/images", StaticFiles(directory=IMG_DIR), name="images")

# DATA MODELS
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    image_url: Optional[str] = None
    transcript_loaded: bool = False

# SESSION STATE 
# keep state in memory. (maybe change to a db)
current_state = {
    "messages": [],
    "youtube_url": None,
    "is_processed": False,
    "image_suggestions": []
}

# ENDPOINTS 

@app.get("/health")
def health_check():
    return {"status": "running"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main Chat Loop:
    1. Receive user message
    2. Run LangGraph
    3. Return last AI message
    """
    global current_state
    
    user_msg = HumanMessage(content=request.message)
    user_input = request.message
    
    # Initialize state if empty
    if not current_state.get("messages"):
        current_state = {
            "messages": [user_msg],
            "youtube_url": None,
            "is_processed": False,
            "image_suggestions": []
        }
    else:
        current_state["messages"].append(user_msg)
        
    # If the user provides a NEW YouTube URL, we must force re-processing.
    if "youtube.com" in request.message or "youtu.be" in request.message:
        current_state["youtube_url"] = request.message
        current_state["is_processed"] = False 
        
        # Clear the previous context immediately to be safe
        from .state import STORE
        STORE.set_df(None)

    # 2. Run Graph
    # use stream() to execute the graph steps
    final_response_text = ""
    try:
        events = graph_app.stream(current_state)
        
        for event in events:
            for key, value in event.items():
                # Update our global state dict with new values
                if "messages" in value:
                    current_state["messages"].extend(value["messages"])
                
                # Update other keys (is_processed, etc)
                for k, v in value.items():
                    if k != "messages":
                        current_state[k] = v
        
        # 3. Extract the last response
        last_msg = current_state["messages"][-1]
        final_response_text = last_msg.content
        
        # Remove unwanted mesage in final output
        stop_words = ["[END_TOOL_RESULT]", "[TOOL_USE]", "User:", "Model:", "<end_of_turn>", "[TOOL_RESULT]"]
        
        for artifact in stop_words:
            final_response_text = final_response_text.replace(artifact, "")
        
        final_response_text = final_response_text.strip()

    except Exception as e:
        print(f"Error processing graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # 4. Check for Generated Images
    # Method A: Check the AI's text
    image_url = None
    if "generated_images/" in final_response_text:
        # extraction logic
        match = re.search(r"generated_images/([\w\d_]+\.png)", final_response_text)
        if match:
            # Convert local path to Server URL
            filename = match.group(1)
            image_url = f"http://localhost:8000/images/{filename}"
            
    # If Method A failed, look at the recent tool outputs
    if not image_url:
        # find the most recent ToolMessage
        for msg in reversed(current_state["messages"]):
            # Stop if we hit a human message (don't search old history)
            if isinstance(msg, HumanMessage): 
                break
                
            # If it's a ToolMessage and contains an image path
            if hasattr(msg, 'content') and "generated_images/" in str(msg.content):
                match = re.search(r"generated_images/([\w\d_]+\.png)", str(msg.content))
                if match:
                    filename = match.group(1)
                    image_url = f"http://localhost:8000/images/{filename}"
                    break

    return ChatResponse(
        response=final_response_text,
        image_url=image_url,
        transcript_loaded=current_state.get("is_processed", False)
    )

@app.post("/clear")
async def clear_history():
    """Reset the session."""
    global current_state
    
    # Clear VectorDB Logic
    STORE.clear()
    
    current_state = {
        "messages": [],
        "youtube_url": None,
        "is_processed": False,
        "image_suggestions": []
    }
    return {"status": "History cleared"}

#  RUNNER
if __name__ == "__main__":
    # Check for Wiki Server file
    if not os.path.exists("wiki_server.py"):
        print("WARNING: 'wiki_server.py' not found in root. MCP tool will fail.")
    
    print("Starting Backend on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)