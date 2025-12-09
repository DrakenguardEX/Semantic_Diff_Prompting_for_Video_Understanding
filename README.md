# Semantic_Diff_Prompting_for_Video_Understanding
Final Project for cs6180 generative AI

## Overview
This project compares baseline frame-by-frame video understanding with semantic diff prompting, which describes only changes between consecutive frames. This approach can reduce token consumption while maintaining important temporal information.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up OpenAI API Key
You need an OpenAI API key to use the vision language model. 

**Get your API key:** https://platform.openai.com/account/api-keys

**Set it as an environment variable:**

On macOS/Linux:
```bash
export OPENAI_API_KEY="sk-your-actual-api-key-here"
```

On Windows:
```bash
set OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Important:** 
- Replace `"sk-your-actual-api-key-here"` with your actual API key from OpenAI
- The key should start with `sk-`
- Make sure there are no extra spaces or quotes around the key when setting it
- If you're using conda/virtualenv, set it in the same terminal session where you run the script

**Verify it's set correctly:**
```bash
echo $OPENAI_API_KEY  # macOS/Linux
echo %OPENAI_API_KEY%  # Windows
```

Alternatively, you can modify `semantic_diff_demo.py` to pass the API key directly:
```python
vlm = VLMClient(api_key="sk-your-actual-api-key-here", model="gpt-4o-mini")
```

## Usage

### Main Demo: Semantic Diff Comparison
Run the main comparison between baseline and diff prompting:

```bash
python semantic_diff_demo.py
```

This will:
1. Load frames from the `test_frame_diff/` folder
2. Run baseline prompting (describes each frame independently)
3. Run semantic diff prompting (describes only changes between frames)
4. Compare outputs and show token statistics

### Simple Vision Test
Test basic image description functionality:

```bash
python vision_test.py
```

This tests a single image (`test_img1.jpg`) with the vision model.

## Project Structure
- `semantic_diff_demo.py` - Main demo comparing baseline vs diff prompting
- `vlm_client.py` - Vision Language Model client wrapper for OpenAI API
- `vision_test.py` - Simple test script for image description
- `test_frame_diff/` - Sample video frames for testing
- `test_img1.jpg` - Sample image for testing

## How It Works
- **Baseline**: Each frame is described independently, leading to redundant information
- **Semantic Diff**: Only changes between consecutive frames are described, reducing token usage while preserving temporal dynamics
