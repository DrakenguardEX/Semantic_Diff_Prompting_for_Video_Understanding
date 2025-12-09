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

The script supports **video files, image folders, and single image files** as input.

#### Using a Video File
```bash
python semantic_diff_demo.py path/to/your/video.mp4
```

**Supported video formats:** `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`, `.webm`, `.m4v`

**Options for video processing:**
```bash
# Limit number of frames processed (useful for testing)
python semantic_diff_demo.py video.mp4 --max-frames 10

# Extract every Nth frame (useful for long videos to reduce processing time)
python semantic_diff_demo.py video.mp4 --frame-interval 5  # Every 5th frame

# Use a different OpenAI model
python semantic_diff_demo.py video.mp4 --model gpt-4o
```

#### Using an Image Folder
```bash
python semantic_diff_demo.py path/to/image/folder
```

Or use the default test folder:
```bash
python semantic_diff_demo.py
# or explicitly:
python semantic_diff_demo.py test_frame_diff
```

#### Using a Single Image File
```bash
python semantic_diff_demo.py path/to/image.jpg
```

#### What the script does:
1. Extracts frames from video (or loads images from folder/file)
2. Runs baseline prompting (describes each frame independently)
3. Runs semantic diff prompting (describes only changes between consecutive frames)
4. Displays comparison results in the terminal
5. Calculates and displays token statistics with reduction percentage
6. **Saves results to a timestamped file** in the `outputs/` directory

**Note:** The script includes a 3-second delay between API calls to prevent rate limiting. Processing many frames may take some time.

### Simple Vision Test
Test basic image description functionality:

```bash
python vision_test.py
```

This tests a single image (`test_img1.jpg`) with the vision model.

## Command-Line Arguments

```
positional arguments:
  input                 Path to video file, image folder, or single image file 
                       (default: test_frame_diff)

optional arguments:
  --max-frames MAX_FRAMES
                        Maximum number of frames to process from video
                        (default: all frames)
  
  --frame-interval FRAME_INTERVAL
                        Extract every Nth frame from video 
                        (1 = all frames, 2 = every other frame, etc.)
                        (default: 1)
  
  --model MODEL         OpenAI model to use 
                        (default: gpt-4o-mini)
```

## Output

### Console Output
The script prints:
- Frame extraction/loading progress
- Side-by-side comparison of baseline vs diff descriptions for each frame
- Token statistics including:
  - Total tokens for baseline approach
  - Total tokens for diff approach
  - Token reduction amount and percentage

### Saved Results
Results are automatically saved to `outputs/results_YYYYMMDD_HHMMSS.txt` with:
- Complete frame-by-frame comparison
- Token statistics
- Timestamp for easy tracking

## Project Structure
- `semantic_diff_demo.py` - Main demo comparing baseline vs diff prompting
  - Supports videos, image folders, and single images
  - Includes rate limiting and error handling
  - Saves results to timestamped files
- `vlm_client.py` - Vision Language Model client wrapper for OpenAI API
  - Handles API calls with automatic retry on rate limits/errors
  - Supports single image and image pair descriptions
- `vision_test.py` - Simple test script for image description
- `test_frame_diff/` - Sample video frames for testing (4 PNG files)
- `test_img1.jpg` - Sample image for testing
- `outputs/` - Directory where results are saved (created automatically)

## How It Works

### Baseline Approach
Each frame is described independently, leading to redundant information being repeated across frames. For example, if a person is walking through a scene, their presence and the background are described in every frame.

### Semantic Diff Approach
Only changes between consecutive frames are described. This means:
- The first frame gets a full description (no previous frame to compare)
- Subsequent frames only describe what changed (movement, new objects, state changes)
- Static elements (background, unchanged objects) are not repeated
- **Result:** Significant token reduction while preserving temporal dynamics

### Example
**Baseline (Frame 2):** "A person is walking on a sidewalk. There are trees in the background. The sky is blue."

**Diff (Frame 2):** "The person has moved forward by two steps. Their right leg is now extended forward."

The diff approach focuses on the change, avoiding repetition of the static scene elements.

## Features

- Video file support (multiple formats)
- Image folder support
- Single image file support
- Frame sampling options (max frames, frame interval)
- Automatic rate limiting (3-second delays)
- Error handling with retries
- Token counting with accurate GPT-4 tokenizer (tiktoken)
- Results saved to timestamped files
- Token reduction statistics

## Requirements

- Python 3.7+
- OpenAI API key
- See `requirements.txt` for Python package dependencies
