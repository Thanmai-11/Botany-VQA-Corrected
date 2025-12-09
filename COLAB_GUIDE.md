# 🚀 Quick Start Guide for Google Colab

## Option 1: Direct Upload to Colab

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)

2. **Upload the Colab Notebook**:
   - Click **File** → **Upload notebook**
   - Upload `generate_dataset_colab.ipynb` from your local machine

3. **Enable GPU**:
   - Click **Runtime** → **Change runtime type**
   - Select **T4 GPU** (free tier)
   - Click **Save**

4. **Run All Cells**:
   - Click **Runtime** → **Run all**
   - Or run cells one by one with `Shift+Enter`

---

## Option 2: Upload Files Manually

If you prefer to upload the Python files manually:

### Step 1: Create New Colab Notebook
- Go to [colab.research.google.com](https://colab.research.google.com)
- Click **File** → **New notebook**

### Step 2: Upload Python Files
Run this in a cell:
```python
from google.colab import files
uploaded = files.upload()
```

Then upload these files:
- `dataset_generator.py`
- `question_templates.py`
- `visual_feature_extractor.py`
- `vqa_validator.py`

### Step 3: Follow the Notebook
Open `generate_dataset_colab.ipynb` and follow the steps!

---

## Option 3: Clone from GitHub (Recommended)

### Step 1: Push to GitHub
From your local machine:
```bash
cd /Users/thanmai/VQA/Botany-VQA-Corrected
git init
git add .
git commit -m "Initial commit: Image-grounded Botany-VQA generator"
git remote add origin https://github.com/yourusername/Botany-VQA-Corrected.git
git push -u origin main
```


### Step 2: Clone in Colab
In a Colab cell:
```python
!git clone https://github.com/yourusername/Botany-VQA-Corrected.git
%cd Botany-VQA-Corrected
```

### Step 3: Run the Notebook
Open `generate_dataset_colab.ipynb` and run!

---

## 📋 Colab Workflow Summary

```
1. Open Colab → Upload notebook
2. Enable GPU (Runtime → Change runtime type → T4 GPU)
3. Run cells sequentially:
   ✓ Install dependencies (2 min)
   ✓ Download Oxford Flowers dataset (5 min)
   ✓ Create labels.json (1 min)
   ✓ Load BLIP-2 model (3 min)
   ✓ Test on sample image (30 sec)
   ✓ Generate pilot (100 images, 20-30 min)
   ✓ Validate pilot (1 min)
   ✓ Generate full dataset (8,189 images, 3-4 hours)
   ✓ Download results
```

---

## ⚠️ Important Colab Tips

### 1. Prevent Timeout
- Keep browser tab **active**
- Click in the notebook occasionally
- Consider **Colab Pro** for longer runtime (up to 24 hours)

### 2. Save Progress to Google Drive
Add this cell early in your notebook:
```python
from google.colab import drive
drive.mount('/content/drive')

# Save checkpoints to Drive
import shutil
shutil.copy('botany_vqa_pilot.csv', '/content/drive/MyDrive/')
```

### 3. Resume if Disconnected
If Colab disconnects, you can resume by:
1. Re-running the setup cells
2. Loading the last saved checkpoint
3. Continuing from where you left off

---

## 🎯 Expected Timeline (with Free Colab GPU)

| Task | Time |
|------|------|
| Setup + Install | 5 min |
| Download Dataset | 5 min |
| Load Model | 3 min |
| Pilot (100 images) | 20-30 min |
| **Full Dataset (8,189 images)** | **3-4 hours** |
| Validation | 2 min |
| **Total** | **~4 hours** |

---

## 💡 Pro Tips

1. **Start with Pilot**: Always run the pilot (100 images) first to verify quality

2. **Monitor GPU Usage**: Check GPU memory with:
   ```python
   !nvidia-smi
   ```

3. **Batch Processing**: The code already processes in batches, but you can adjust batch size if needed

4. **Save Frequently**: Save intermediate results to Google Drive every 1,000 images

5. **Use Colab Pro**: For the full dataset, consider Colab Pro ($10/month) for:
   - Longer runtime (24 hours vs 12 hours)
   - Better GPUs (A100, V100)
   - Faster processing

---

## 📥 What You'll Download

After completion, you'll have:

1. **botany_vqa_grounded.csv** (~15 MB)
   - 81,890 QA pairs
   - Image-grounded answers
   - 4 difficulty levels

2. **dataset_statistics.json** (~2 KB)
   - Dataset metrics
   - Question type distribution
   - Quality scores

3. **validation_report.txt** (~1 KB)
   - Consistency checks
   - Accuracy metrics
   - Quality assessment

---

## 🆘 Troubleshooting

### "No GPU available"
- Make sure you enabled GPU runtime
- Try: Runtime → Change runtime type → T4 GPU → Save

### "Session timeout"
- Keep browser active
- Click in notebook occasionally
- Consider Colab Pro

### "Out of memory"
- Restart runtime: Runtime → Restart runtime
- Use smaller batch size in code

### "Model download failed"
- Check internet connection
- Retry the cell
- Model is ~5GB, may take a few minutes

---

## 🎓 Ready to Start?

1. Open the Colab notebook: `generate_dataset_colab.ipynb`
2. Enable GPU
3. Run all cells
4. Wait 3-4 hours
5. Download your corrected dataset!

**Good luck with your research! 🌸**
