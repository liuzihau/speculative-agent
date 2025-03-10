import requests
import pickle
import uuid
import uvicorn
import nest_asyncio
from transformers import AutoTokenizer
from fastapi import FastAPI, Request, Response
from pyngrok import ngrok
import threading

# Draft Server ngrok URL (Update this with your actual draft server URL)
DRAFT_SERVER_URL = "https://a86a-34-125-120-118.ngrok-free.app/predict"# Enable FastAPI to run in Colab

nest_asyncio.apply()

app = FastAPI()
task_results = {}  # Store received responses

# Load GPT-2 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Start ngrok to expose the request server
port = 7000  # Change from 6000 to 7000
public_url = ngrok.connect(port).public_url
print(f"🚀 Request Server is live at: {public_url}/receive_result")

@app.post("/send_task")
async def send_task():
    """Generate a task, send it to the Draft Server, and track its result"""

    # Generate a unique task ID
    task_id = str(uuid.uuid4())

    # Input text
    input_text = "Hello, how are you?"

    # Tokenize input text to get token IDs
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids  # Keep as tensor

    # Prepare the request payload
    input_data = {
        "task_id": task_id,
        "ids": input_ids,  # Send tokenized input instead of raw text
        "request_server_url": f"{public_url}/receive_result"  # Send results back here
    }

    # Send task to the Draft Server
    response = requests.post(DRAFT_SERVER_URL, data=pickle.dumps(input_data))

    # Handle response
    if response.status_code == 200:
        task_info = pickle.loads(response.content)
        print(f"✅ Task submitted! Task ID: {task_id}, Queue Length: {task_info['queue_tasks']}")
        return Response(content=pickle.dumps(task_info), media_type="application/octet-stream")
    else:
        print(f"❌ Error submitting task: {response.status_code}")
        print(response.text)
        return Response(content=pickle.dumps({"error": response.text}), media_type="application/octet-stream")

@app.post("/receive_result")
async def receive_result(request: Request):
    """Receive the final result from the Draft Server"""
    response_pickle = await request.body()
    response_data = pickle.loads(response_pickle)

    task_id = response_data["task_id"]
    task_results[task_id] = response_data["response"]

    print(f"✅ Received result for Task ID {task_id}:")
    print("🔹 Generated Tokens:", response_data["response"]["generated_tokens"])
    print("🔹 Logits Shape:", len(response_data["response"]["logits"]), "tokens x vocab size")

    return Response(content=pickle.dumps({"acknowledged": True}), media_type="application/octet-stream")

# Function to send a task after the server starts
def send_task_after_startup():
    import time
    time.sleep(3)  # Wait 3 seconds for the server to be fully up

    REQUEST_SERVER_URL = f"{public_url}/send_task"
    response = requests.post(REQUEST_SERVER_URL)

    if response.status_code == 200:
        print("✅ Automatically sent a task after server startup!")
    else:
        print("❌ Failed to send task:", response.text)

# Run the function in a separate thread
threading.Thread(target=send_task_after_startup).start()

# Run the Request Server
uvicorn.run(app, host="0.0.0.0", port=port)