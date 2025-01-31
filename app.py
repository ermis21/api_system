from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
from contextlib import asynccontextmanager
import torch
import numpy as np
from datetime import datetime, timedelta
import threading
import time
from typing import Dict, List
import uuid
from threading import Lock
from huggingface_hub import login
import os
token = os.getenv("HUGGING_FACE_HUB_TOKEN")
login(token=token)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    unload_model()

app = FastAPI(lifespan=lifespan)
inactivity_thr = 60

# Request models
class LoadModelRequest(BaseModel):
    model_name: str = "gpt2"

class ChangeModelRequest(BaseModel):
    new_model_name: str

class InferenceRequest(BaseModel):
    text: str
    max_new_tokens: int = 1000
    temperature: float = 0.7
    repetition_penalty: float = 1.1
    session_id: str

class ChangeInactivityThresholdRequest(BaseModel):
    new_threshold: int

class SystemPromptRequest(BaseModel):
    prompt: str

# Global variables
system_prompt = "You are a helpful assistant."
tokenizer = None
current_model = None
model_pipeline = None
last_request_time = datetime.now()
vram_lock = threading.Lock()
timer = None

# Session management
session_histories: Dict[str, List[dict]] = {}
session_lock = Lock()
session_system_prompts: Dict[str, str] = {}

def get_or_create_session(session_id: str):
    with session_lock:
        if session_id not in session_histories:
            # Initialize new session with system prompt
            session_histories[session_id] = [
                {"role": "system", "content": system_prompt}
            ]
            session_system_prompts[session_id] = system_prompt
        return session_histories[session_id]

def unload_model():
    global model_pipeline, current_model
    with vram_lock:
        if model_pipeline is not None:
            del model_pipeline
            del current_model
            torch.cuda.empty_cache()
            model_pipeline = None
            current_model = None
            print("Model unloaded from VRAM")

def check_inactivity():
    global last_request_time, timer
    while True:
        time.sleep(60)
        if (datetime.now() - last_request_time).seconds > inactivity_thr:
            print(f"Inactivity period reached ({inactivity_thr}), unloading model")
            unload_model()
            timer = None
            break

def get_free_vram():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)  # GPU index 0
    info = nvmlDeviceGetMemoryInfo(handle)
    nvmlShutdown()
    return info.free / (1024 ** 3)  # Convert bytes to GB

from huggingface_hub import HfApi

def get_model_size(repo_id, branch="main"):
    api = HfApi()
    files = api.list_repo_tree(repo_id=repo_id, 
                               revision=branch, 
                               token=token, 
                               recursive=True)

    # Sum sizes only for files (not folders)
    total_size = sum(file.size for file in files if hasattr(file, "size") and ".safetensors" in file.path )
    
    return total_size / (1024**3)  # Convert bytes to GB

@app.post("/load_model")
async def load_model(request: LoadModelRequest = Body(...)):
    global model_pipeline, current_model, timer, last_request_time
    
    last_request_time = datetime.now()
    
    free_vram = np.around(get_free_vram(),decimals=0)
    model_ram = get_model_size(request.model_name)*2 # Empiricaly desided 200%
    if model_ram > free_vram: 
        unload_model()
        return {"State": f"The model you requested does not fit in VRAM (Estimated size {model_ram:.2f}GB>{free_vram:.2f}GB)"}
    
    if model_pipeline is not None:
        unload_model()

    try:
        tokenizer = AutoTokenizer.from_pretrained(request.model_name)
        model = AutoModelForCausalLM.from_pretrained(request.model_name)#, trust_remote_code = True)
        
        model_config = model.config.to_dict()
        supports_flash_attention = "flash_attention_2" in model_config
        pipeline_kwargs = {"model": model, "tokenizer": tokenizer}
        if supports_flash_attention:
            pipeline_kwargs["flash_attention_2"] = True
        pipeline_kwargs["device"] = 0 if torch.cuda.is_available() else -1
        model_pipeline = pipeline("text-generation", **pipeline_kwargs, temperature = 0.1)

        current_model = request.model_name
    except Exception as e:
        unload_model()
        raise HTTPException(status_code=400, detail=f"Error loading model: {str(e)}")
    
    if timer is None:
        timer = threading.Thread(target=check_inactivity, daemon=True)
        timer.start()
    
    return {"status": f"Model {request.model_name} loaded successfully"}

@app.post("/change_model")
async def change_model(request: ChangeModelRequest = Body(...)):

    free_vram = np.around(get_free_vram(),decimals=0)
    model_ram = get_model_size(request.new_model_name)*2 # Empiricaly desided 200%
    if model_ram > free_vram: 
        unload_model()
        return {"State": f"The model you requested does not fit in VRAM (Estimated size {model_ram:.2f}GB>{free_vram:.2f}GB)"}

    await load_model(LoadModelRequest(model_name=request.new_model_name))
    return {"status": f"Changed to model {request.new_model_name} successfully ({model_ram:.2f}GB)"}

@app.post("/infer")
async def infer(request: InferenceRequest = Body(...)):
    global last_request_time
    
    if model_pipeline is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    last_request_time = datetime.now()
    
    try:
        result = model_pipeline(request.text)[0]['generated_text']
        return {"input": request.text, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/system_prompt")
async def update_system_prompt(request: SystemPromptRequest = Body(...)):
    global system_prompt
    system_prompt = request.prompt
    return {"message": "System prompt updated (will affect new sessions only)"}

@app.post("/chat")
async def infer(request: InferenceRequest = Body(...)):
    global model_pipeline, tokenizer
    
    if None in (model_pipeline, tokenizer):
        raise HTTPException(400, "Model/tokenizer not loaded")
    
    try:
        # Get or create session history
        history = get_or_create_session(request.session_id)
        
        # Add user message to history
        history.append({"role": "user", "content": request.text})
        
        # Format chat with full history
        formatted_input = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate response
        result = model_pipeline(formatted_input)[0]['generated_text']
        
        # Add assistant response to history
        history.append({"role": "assistant", "content": result})
        
        return {
            "session_id": request.session_id,
            "result": result,
            "full_history": history
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    
@app.post("/change_inactivity_threshold")
async def change_inactivity_threshold(request: ChangeInactivityThresholdRequest = Body(...)):
    global inactivity_thr
    inactivity_thr = request.new_threshold
    return {"status": f"Inactivity threshold changed to {inactivity_thr} seconds"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)