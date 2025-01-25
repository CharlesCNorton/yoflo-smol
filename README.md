--------------------------------------------------------------------------------
SMOL-YOFLO (v1.0.0)
By: Charles C. Norton

INTRODUCTION
------------
SMOL-YOFLO is a minimal, SmolVLM-based CLI application derived from the YO-FLO project. While YO-FLO harnesses Microsoft’s Florence-2 for object detection, SMOL-YOFLO focuses on direct vision-language question answering and yes/no inference—without bounding boxes.

This streamlined approach offers a lighter-weight alternative ideal for simpler VQA tasks or hardware-constrained environments, while preserving the YO-FLO structure, CLI workflow, logging, and recording options.

TABLE OF CONTENTS
-----------------
1) INTRODUCTION  
2) DIFFERENCES FROM YO-FLO  
3) WHAT IS SMOL-YOFLO?  
4) CORE FUNCTIONALITY  
5) KEY FEATURES  
6) USAGE AND DEMO  
7) INSTALLATION AND SYSTEM REQUIREMENTS  
8) COMMAND-LINE FLAGS  
9) EXAMPLE COMMANDS  
10) INFERENCE CHAINS  
11) PERFORMANCE TUNING AND OPTIMIZATION  
12) TROUBLESHOOTING AND FAQ  
13) FUTURE DIRECTIONS  
14) LICENSE AND ACKNOWLEDGMENTS  

1) INTRODUCTION
---------------
SMOL-YOFLO uses smaller SmolVLM models (such as 256M or 500M parameters) instead of the large Florence-2 backbone. Users pose questions like "Is the person wearing a hat?" or "Describe the scene," receiving free-form text answers rather than bounding boxes. The result is a simpler tool suited to general VQA or captioning scenarios.

2) DIFFERENCES FROM YO-FLO
--------------------------
- No bounding-box detection or object tracking.  
- Runs on SmolVLM models, smaller than Florence-2.  
- Generates text answers only, including yes/no responses to binary questions.  
- Maintains YO-FLO's CLI structure, threads, screenshot, and logging features.

3) WHAT IS SMOL-YOFLO?
----------------------
SMOL-YOFLO stands for "Smol Vision-Language YO-FLO." It is a command-line interface that takes live camera feeds or RTSP streams, applies a SmolVLM model for vision-language inference, and logs or displays textual outputs. Use it to ask questions about a scene or to generate short captions.

4) CORE FUNCTIONALITY
---------------------
• Text-based vision Q&A or captioning  
• Yes/No inference for quick scenario checks  
• Inference chain logic for multiple sequential questions  
• Screenshot capture and logging  
• Video recording triggers based on "yes" or "no"  
• Optional headless operation for server or cloud use

5) KEY FEATURES
---------------
• Supports smaller SmolVLM models for reduced memory usage  
• Recording triggered by yes/no answers  
• Multicamera or multistream support  
• Headless mode for performance gains  
• Inference rate display (inferences per second)  
• Screenshot and logs for record-keeping

6) USAGE AND DEMO
-----------------
SMOL-YOFLO works similarly to YO-FLO. You specify the input source (webcam index or RTSP URL) and a text prompt or chain of prompts. The model produces direct text answers. You can enable screenshot capture, headless mode, or logging from the command line. 

7) INSTALLATION AND SYSTEM REQUIREMENTS
---------------------------------------
- Python 3.10+ recommended  
- PyTorch, transformers, OpenCV, Pillow, and huggingface-hub for model I/O and inference  
- GPU strongly recommended for real-time usage  
- CUDA 11.7 or newer if doing GPU half-precision or flash-attention

To install from source:
1) Clone or download the smol-yoflo repository  
2) Run pip install .  
3) Confirm installation by typing python smol_yoflo.py --help

8) COMMAND-LINE FLAGS
---------------------
• -ph "Question" : A single text or yes/no query  
• -ic "Q1" "Q2" : Multiple yes/no questions for an inference chain  
• -hl : Headless mode (no GUI)  
• -lf : Write alerts to a file (alerts.log)  
• -ir : Show inferences per second  
• -pp : Pretty print results  
• -r infy/infn : Video record triggers on yes/no  
• -mp /path/to/model : Use a local SmolVLM model  
• -dm : Download default SmolVLM from Hugging Face  
• -4bit : Enable 4-bit quantization

9) EXAMPLE COMMANDS
-------------------
• Single yes/no question using webcam 0, auto-download model:
  python smol_yoflo.py -dm -ph "Is the person smiling?" -wi 0

• Multiple questions in a chain:
  python smol_yoflo.py -mp /path/to/SmolVLM -ic "Is there a cat?" "Is it sitting down?" -wi 0

• Recording triggered by "yes" answers, headless:
  python smol_yoflo.py -dm -ph "Is the door open?" -r infy -hl

10) INFERENCE CHAINS
--------------------
You can pass multiple phrases to -ic. SMOL-YOFLO will ask each question in turn, parse answers for "yes" or "no," and finalize a pass/fail if the majority are "yes." This is useful for multi-step conditions without needing separate runs.

11) TROUBLESHOOTING AND FAQ
---------------------------
• "Model not found" => Check -mp path or confirm download.  
• Slow performance => Reduce resolution, confirm half precision, or reduce number of tokens.  
• "Why no bounding boxes?" => SMOL-YOFLO does not support detection; see YO-FLO for that functionality.

12) FUTURE DIRECTIONS
---------------------
• Multi-turn chat context (storing conversation history)  
• Automatic advanced prompt templates  
• Broader range of SmolVLM model sizes (256M, 1.7B, etc.)  
• Potential integration with custom local retrieval or embeddings
