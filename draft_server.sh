#!/bin/bash

export CUDA_VISIBLE_DEVICES=1  # Assign second GPU
PORT=5001

python3 my_llm_server.py --port $PORT --gpu 1