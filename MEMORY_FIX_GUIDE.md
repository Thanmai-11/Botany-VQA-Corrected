# 🔧 Memory Fix Guide for Colab

## Problem
Getting `CUDA out of memory` error when running BLIP-2 on Colab's T4 GPU (15GB).

## Solution Applied
We've added **8-bit quantization** and **GPU memory management** to reduce memory usage from ~14GB to ~4-6GB.

---

## 📋 Steps to Fix in Your Colab Notebook

### 1. **Pull Latest Changes**
In your Colab notebook, run this cell:

```python
# Pull latest changes from GitHub
!git pull origin main
```

### 2. **Restart Runtime**
- Click **Runtime** → **Restart runtime**
- This clears all GPU memory

### 3. **Install Additional Dependency**
Add this to your dependencies cell (Step 2):

```python
# Install required packages (UPDATED - added bitsandbytes for 8-bit)
!pip install -q transformers accelerate bitsandbytes scikit-learn opencv-python
print("✓ Dependencies installed!")
```

### 4. **Load Model with Memory Optimization**
Update Step 6 to use 8-bit quantization:

```python
from dataset_generator import BotanyVQAGenerator

# Initialize generator with 8-bit quantization (MEMORY OPTIMIZED)
generator = BotanyVQAGenerator(
    model_name="Salesforce/blip2-opt-2.7b",
    device="cuda",
    use_8bit=True  # ← This reduces memory by ~60%!
)

print("✓ Model loaded on GPU with 8-bit quantization!")
```

### 5. **Test Again**
Run Step 7 (Test on Sample Image) - it should work now! 🎉

---

## 🧠 What Changed?

### Memory Optimizations:
1. **8-bit Quantization**: Reduces model size from ~14GB to ~5GB
2. **Automatic Device Mapping**: Efficiently distributes model across GPU
3. **Gradient Checkpointing**: Saves memory during inference
4. **GPU Cache Clearing**: Prevents memory accumulation during batch processing

### Expected Memory Usage:
- **Before**: ~14GB (causing OOM)
- **After**: ~4-6GB (plenty of headroom)

---

## 🚨 If Still Getting OOM Errors

### Option 1: Use Smaller Model
```python
generator = BotanyVQAGenerator(
    model_name="Salesforce/blip2-opt-2.7b",  # Current: 2.7B params
    # Try: "Salesforce/blip2-flan-t5-base"  # Smaller alternative
    device="cuda",
    use_8bit=True
)
```

### Option 2: Reduce Beam Search
In `dataset_generator.py`, line 98, reduce `num_beams`:
```python
outputs = self.model.generate(
    **inputs, 
    max_new_tokens=50,
    min_length=1,
    num_beams=3,  # ← Reduce from 5 to 3
    temperature=1.0
)
```

### Option 3: Process in Smaller Batches
When generating dataset, process fewer images at a time:
```python
# Instead of 100 images at once
pilot_df = generator.generate_dataset(
    image_dir="oxford_flowers_102/jpg",
    labels_file="oxford_flowers_102/labels.json",
    output_csv="botany_vqa_pilot.csv",
    num_images=50,  # ← Reduce from 100 to 50
    qa_per_image=10
)
```

---

## ✅ Verification

After loading the model, you should see output like:
```
Using device: cuda
GPU Memory before loading: 0.00 GB
Loading model: Salesforce/blip2-opt-2.7b...
8-bit quantization: True
Model loaded successfully!
GPU Memory after loading: 4.52 GB
GPU Memory reserved: 4.89 GB
```

If you see **~4-6GB** instead of ~14GB, you're good to go! 🚀

---

## 📊 Memory Monitoring

Add this cell to monitor GPU memory during processing:

```python
import torch

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {allocated:.2f}GB / {total:.2f}GB ({allocated/total*100:.1f}%)")

# Call this anytime to check memory
print_gpu_memory()
```

---

## 🎯 Summary

✅ **Pull latest code**: `!git pull origin main`  
✅ **Restart runtime**: Clear all memory  
✅ **Install bitsandbytes**: For 8-bit support  
✅ **Use `use_8bit=True`**: When loading model  
✅ **Test**: Should work with ~4-6GB instead of ~14GB  

You're all set! 🌸
