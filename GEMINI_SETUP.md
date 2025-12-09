# 🌸 Gemini-Powered VQA Generation Guide

Use Google's advanced Gemini models to generate your Botany VQA dataset. This method requires **NO GPU** and produces higher quality answers.

## 🚀 Step 1: Get Your API Key
1. Go to [Google AI Studio](https://aistudio.google.com/).
2. Click **"Get API key"** (top left).
3. Click **"Create API key"**.
4. Copy the key string (starts with `AIza...`).

## 📦 Step 2: Install Libraries in Colab
Run this in a new cell:
```python
!pip install -q google-generativeai tqdm pandas pillow
```

## 💎 Step 3: Run the Generator
1. Upload `gemini_generator.py` to Colab (along with `question_templates.py`, `dataset_generator.py`, etc.).
2. Run this code block:

```python
import os
from gemini_generator import GeminiVQAGenerator

# Paste your API key here (or use secrets)
API_KEY = "PASTE_YOUR_API_KEY_HERE"

# Initialize generator (Use 'gemini-1.5-flash' for speed, 'gemini-1.5-pro' for quality)
generator = GeminiVQAGenerator(
    api_key=API_KEY, 
    model_name="gemini-1.5-flash"
)

# Generate Dataset
df = generator.generate_dataset(
    image_dir="oxford_flowers_102/jpg",
    labels_file="oxford_flowers_102/labels.json",
    output_csv="botany_vqa_gemini.csv",
    num_images=100,  # Pilot run
    qa_per_image=10
)

print("✅ Generation Complete!")
```

## ❓ FAQ
**Q: Which model should I use?**
- **Flash**: Fast, cheap/free capable. Good for large datasets.
- **Pro**: Smarter, better at detailed descriptions. Slower rate limits on free tier.

**Q: Is it free?**
- Yes, the "Free of Charge" tier allows 15 requests per minute (Flash) or 2 RPM (Pro).
- With **Google AI Premium**, limits are much higher.
