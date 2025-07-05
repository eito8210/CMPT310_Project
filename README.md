# 🚀 CMPT310 Project - Environment Setup

## Prerequisites
- Python 3.8+ 
- Git
- 5GB free disk space

## Setup Steps

### 1. Navigate to Project Directory
```bash
cd CMPT310_PROJECT
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv smile_detection_env

# Activate it
# Mac/Linux:
source smile_detection_env/bin/activate

# Windows:
smile_detection_env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Setup
```bash
python test_setup.py
```

Expected output:
```
✅ os
✅ numpy
✅ matplotlib.pyplot
✅ sklearn.model_selection
✅ cv2
✅ tqdm
✅ keras (TensorFlow)
🎉 Environment setup successful!
```

### 5. Test Data Processing
```bash
python data_preprocessing.py
# Choose option 1: Team Zip Files
```

## Daily Usage

```bash
# Activate environment (do this every time)
source smile_detection_env/bin/activate  # Mac/Linux
# smile_detection_env\Scripts\activate   # Windows

# Run scripts
python data_preprocessing.py
python integrated_training.py
python integrated_video_tracker.py
```

## Troubleshooting

### Virtual environment issues:
```bash
rm -rf smile_detection_env/
python -m venv smile_detection_env
source smile_detection_env/bin/activate
pip install -r requirements.txt
```

### Missing packages:
```bash
# Make sure environment is activated first
source smile_detection_env/bin/activate
pip install package_name
```

That's it! 🎉