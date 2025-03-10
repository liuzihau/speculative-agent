import json
import requests
import pickle
import torch
import uvicorn
import asyncio
from fastapi import FastAPI, Request, Response, BackgroundTasks
from transformers import AutoModelForCausalLM, AutoTokenizer
import nest_asyncio
from pyngrok import ngrok
from contextlib import asynccontextmanager


# Apply nest_asyncio to allow FastAPI to run in Colab
nest_asyncio.apply()

# Load configuration manually (Colab does not support file paths easily)
CONFIG = {
    "model_name": "gpt2",
    "draft_length": 8,  # Max length for generated text,
    "device": "cuda:0"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load LLM Model on Startup"""
    global model, tokenizer
    model_name = CONFIG["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(CONFIG["device"])
    print("Draft Server running LLM Model on GPU" if torch.cuda.is_available() else "CPU")
    yield  # Keep the app running

async def process_request(inputs, request_server_url):
    """Process input and send response back to the request server"""
    try:
        task_id = inputs["task_id"]  # Extract task_id from request server
        # input_ids = tokenizer(inputs["ids"], return_tensors="pt").input_ids.to(model.device)

        outputs = model.generate(
            inputs["ids"].to(CONFIG["device"]),
            max_length=CONFIG["draft_length"],
            return_dict_in_generate=True,
            output_scores=True,
        )
        
        logits = torch.stack(outputs.scores, dim=1)  # Shape: (batch_size, sequence_length, vocab_size)

        response_data = {
            "task_id": task_id,
            "queue_tasks": task_queue.qsize(),
            "response": {
                "logits": logits.tolist(),  # Convert to list for serialization
                "generated_tokens": outputs.sequences.tolist()
            }
        }

        requests.post(request_server_url, data=pickle.dumps(response_data))  # Send final result
    finally:
        task_queue.task_done()  # Mark task as completed

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(request: Request, background_tasks: BackgroundTasks):
    """Queue the task and return the task_id with queue length"""
    inputs_pickle = await request.body()
    inputs = pickle.loads(inputs_pickle)

    request_server_url = inputs["request_server_url"]  # Extract URL from request

    # Add task to queue
    task_queue.put_nowait((inputs, request_server_url))

    # Process tasks sequentially
    if task_queue.qsize() == 1:  # Start processing if it's the first task
        background_tasks.add_task(task_worker)

    return Response(content=pickle.dumps({"task_id": inputs["task_id"], "queue_tasks": task_queue.qsize()}), media_type="application/octet-stream")

async def task_worker():
    """Worker that processes tasks sequentially"""
    while not task_queue.empty():
        inputs, request_server_url = await task_queue.get()
        await process_request(inputs, request_server_url)

# Run the FastAPI app using uvicorn with ngrok

task_queue = asyncio.Queue()  # Async queue to handle multiple tasks

port = 5001
public_url = ngrok.connect(port).public_url
print(f"Draft Server is live at: {public_url}")

uvicorn.run(app, host="0.0.0.0", port=port)