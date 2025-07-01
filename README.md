# CMPT310_Project

# Step-by-Step Project Cleanup Guide

## 🎯 Goal
Clean up your project to include only source code in GitHub, while preserving large files for cloud storage sharing.

## 📋 Current File Analysis

### ✅ Keep in GitHub (Source Code)
```
📄 Python Files:
├── data_preprocessing.py          # Eito's data pipeline
├── integrated_training.py         # Seungyeop's training
├── integrated_video_tracker.py    # Jake's video processing  
├── smile_video_tracker.py         # Alternative video processor
├── train_smile_resnet.py          # Basic training script
├── test_setup.py                  # Environment verification
└── download_data.py               # Auto-download script

📄 Configuration Files:
├── requirements.txt               # Dependencies
├── README.md                      # Documentation
├── data_config.json              # Data configuration
└── .gitignore                     # Git exclusion rules
```

### 🚫 Exclude from GitHub (Large Files)
```
📦 Large Data Files:
├── smile_dataset/                 # ~1-2GB dataset directory
├── FERPlus/                       # ~500MB FER+ dataset
├── positives.zip                  # ~150MB smile images
├── negatives.zip                  # ~120MB no-smile images
├── smile_resnet18.pt             # ~45MB trained model
├── Smiling_Video.mp4             # ~50MB test video
└── *.npz files                   # Processed datasets

📁 Generated Directories:
├── __pycache__/                  # Python cache
├── .vscode/                      # IDE settings
└── temp/logs/                    # Temporary files
```

## 🛠️ Cleanup Steps

### Step 1: Run Cleanup Script
```bash
# Download and run the cleanup script
bash cleanup_script.sh

# Or manually check file sizes
find . -type f -size +10M -not -path "./.git/*"
```

### Step 2: Verify .gitignore
Ensure your `.gitignore` contains:
```gitignore
# Large data files
smile_dataset/
FERPlus/
*.zip
*.pt
*.npz
*.mp4
*.avi

# Python cache
__pycache__/
*.pyc

# IDE files
.vscode/
.idea/
```

### Step 3: Upload Large Files to Cloud Storage

#### Option A: Google Drive
1. Create folder: "CMPT310_Project_Data"
2. Upload files:
   - `positives.zip`
   - `negatives.zip` 
   - `smile_resnet18.pt`
   - `Smiling_Video.mp4`
3. Get shareable links
4. Update `download_data.py` with file IDs

#### Option B: Temporary Backup
```bash
# Create backup directory
mkdir ../CMPT310_BACKUP

# Move large files to backup
mv smile_dataset ../CMPT310_BACKUP/
mv FERPlus ../CMPT310_BACKUP/
mv *.zip ../CMPT310_BACKUP/
mv *.pt ../CMPT310_BACKUP/
mv *.mp4 ../CMPT310_BACKUP/
mv *.npz ../CMPT310_BACKUP/
```

### Step 4: Git Operations
```bash
# Initialize Git (if not done)
git init

# Add only source code files
git add *.py
git add *.txt
git add *.md
git add *.json
git add .gitignore

# Check what will be committed
git status

# Commit
git commit -m "Initial commit: Smile detection source code

Features:
- Data preprocessing pipeline (Eito)
- ResNet18 training system (Seungyeop)  
- Real-time video processing (Jake)
- Automated data download system
- Team collaboration tools

Large data files managed via cloud storage."

# Add remote and push
git remote add origin https://github.com/your-username/CMPT310_PROJECT.git
git push -u origin main
```

## 🎯 Final Project Structure

### In GitHub Repository:
```
CMPT310_PROJECT/
├── 📄 Source Code
│   ├── data_preprocessing.py      # 15KB
│   ├── integrated_training.py     # 8KB
│   ├── integrated_video_tracker.py # 12KB
│   ├── smile_video_tracker.py     # 10KB
│   ├── train_smile_resnet.py      # 5KB
│   ├── test_setup.py              # 3KB
│   └── download_data.py           # 18KB
│
├── 📄 Configuration
│   ├── requirements.txt           # 1KB
│   ├── README.md                  # 25KB
│   ├── data_config.json           # 5KB
│   └── .gitignore                 # 3KB
│
└── 📄 Total Repository Size: ~100KB ✅
```

### In Cloud Storage:
```
Google Drive: CMPT310_Project_Data/
├── positives.zip                  # 150MB
├── negatives.zip                  # 120MB
├── smile_resnet18.pt             # 45MB
├── Smiling_Video.mp4             # 50MB
└── team_dataset.npz              # 200MB

Total Cloud Storage: ~565MB
```

## 🚀 Team Usage After Cleanup

### For New Team Members:
```bash
# 1. Clone repository (fast - only source code)
git clone https://github.com/your-username/CMPT310_PROJECT.git
cd CMPT310_PROJECT

# 2. Install dependencies  
pip install -r requirements.txt

# 3. Download data files automatically
python download_data.py

# 4. Verify setup
python test_setup.py

# 5. Start development
python data_preprocessing.py
```

### For Existing Team Members:
```bash
# Backup your current work
cp -r . ../CMPT310_BACKUP_$(date +%Y%m%d)

# Pull latest changes
git pull origin main

# Restore your data files
python download_data.py
# Or manually restore from backup
```

## ✅ Benefits of This Approach

1. **Fast Git Operations** - Only ~100KB in repository
2. **No Size Limits** - Large files in cloud storage
3. **Easy Team Setup** - One command downloads everything
4. **Version Control** - Source code properly tracked
5. **Professional Structure** - Industry standard approach

## 🚨 Important Notes

- **Never commit large files to Git** - Repository becomes slow
- **Always backup before cleanup** - Protect your work
- **Test download script** - Ensure team can get files
- **Update cloud URLs** - Keep download links current
- **Document changes** - Update README with setup instructions

## 🆘 If Something Goes Wrong

### Restore from backup:
```bash
# If you created backup
cp -r ../CMPT310_BACKUP/* .

# Or restore specific files
cp ../CMPT310_BACKUP/smile_resnet18.pt .
cp -r ../CMPT310_BACKUP/smile_dataset .
```

### Reset Git history:
```bash
# If you accidentally committed large files
git reset --hard HEAD~1  # Remove last commit
# Or start fresh:
rm -rf .git
git init
```

Ready to clean up? Start with the cleanup script! 🧹