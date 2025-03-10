import json
import requests
import argparse
import pickle
from fastapi import FastAPI, Request, Response, BackgroundTasks
import uvicorn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import asyncio

# Load configuration
CONFIG_PATH = "./config.json"
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.loads(f.read())

app = FastAPI()
task_queue = asyncio.Queue()  # Async queue to handle multiple tasks

@app.on_event("startup")
def load_model():
    """Load LLM Model on Startup"""
    global model, tokenizer
    model_name = CONFIG["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(f"cuda:{args.gpu}")
    print(f"Server running LLM Model on GPU {args.gpu}")

async def process_request(inputs, request_server_url):
    """Process input and send response back to the request server"""
    try:
        task_id = inputs["task_id"]  # Extract task_id from request server
        input_ids = tokenizer(inputs["ids"], return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(
                    input_ids,
                    max_length=CONFIG["draft_length"],
                    return_dict_in_generate=True,  # Return a dictionary with details
                    output_scores=True,  # Include logits
                    )
        logits = torch.stack(outputs.scores, dim=1)  # Shape: (batch_size, sequence_length, vocab_size)

        response_data = {
            "task_id": task_id, 
            "queue_tasks": task_queue.qsize(), 
            "response": {
                "logits":logits, 
                "generated_tokens":outputs.sequences
                }
            }
        requests.post(request_server_url, data=pickle.dumps(response_data))  # Send final result
    finally:
        task_queue.task_done()  # Mark task as completed

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)  # Ensure correct GPU allocation
    uvicorn.run(app, host="0.0.0.0", port=args.port)
