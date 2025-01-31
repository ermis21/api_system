# FastAPI Model Inference API Manual

## Overview
This API allows users to load, change, and interact with language models from the Hugging Face Model Hub. The API supports text inference, session-based chat, and inactivity-based model unloading.

## Authentication
The API uses a Hugging Face Hub token for authentication when loading models. Ensure the environment variable `HUGGING_FACE_HUB_TOKEN` is set before running the API.

## Base URL
```
http://<host>:8000
```

## Endpoints

### 1. Load a Model
**Endpoint:**
```
POST /load_model
```
**Description:** Loads a specified model into memory.
**Request Body:**
```json
{
  "model_name": "gpt2"
}
```
**Response:**
```json
{
  "status": "Model gpt2 loaded successfully"
}
```

### 2. Change the Current Model
**Endpoint:**
```
POST /change_model
```
**Description:** Changes the loaded model to a new one.
**Request Body:**
```json
{
  "new_model_name": "new_model"
}
```
**Response:**
```json
{
  "status": "Changed to model new_model successfully (size in GB)"
}
```

### 3. Text Inference
**Endpoint:**
```
POST /infer
```
**Description:** Generates text based on user input.
**Request Body:**
```json
{
  "text": "Hello, how are you?",
  "max_new_tokens": 100,
  "temperature": 0.7,
  "repetition_penalty": 1.1,
  "session_id": "12345"
}
```
**Response:**
```json
{
  "input": "Hello, how are you?",
  "result": "I am fine, thank you!"
}
```

### 4. Chat with Context
**Endpoint:**
```
POST /chat
```
**Description:** Maintains chat sessions and generates responses based on conversation history.
**Request Body:**
```json
{
  "text": "What's your name?",
  "session_id": "abc123"
}
```
**Response:**
```json
{
  "session_id": "abc123",
  "result": "I am a virtual assistant!",
  "full_history": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's your name?"},
    {"role": "assistant", "content": "I am a virtual assistant!"}
  ]
}
```

### 5. Update System Prompt
**Endpoint:**
```
POST /system_prompt
```
**Description:** Updates the system prompt for future sessions.
**Request Body:**
```json
{
  "prompt": "You are a knowledgeable tutor."
}
```
**Response:**
```json
{
  "message": "System prompt updated (will affect new sessions only)"
}
```

### 6. Change Inactivity Threshold
**Endpoint:**
```
POST /change_inactivity_threshold
```
**Description:** Updates the inactivity time before the model is unloaded.
**Request Body:**
```json
{
  "new_threshold": 120
}
```
**Response:**
```json
{
  "status": "Inactivity threshold changed to 120 seconds"
}
```

## Model Management
- The system automatically checks VRAM before loading a model.
- Models are unloaded if inactivity exceeds the threshold.
- Flash attention is enabled if the model supports it.

## Notes
- Ensure `HUGGING_FACE_HUB_TOKEN` is set before running the API.
- The API runs on port 8000 by default.
- Use valid Hugging Face model names to avoid loading errors.

## License
This API is open-source and can be modified as needed.

