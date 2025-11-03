# GitHub Setup Guide

Follow these steps to push this repository to GitHub and use it in Google Colab.

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `learning_to_see`
3. Description: "Vision-language learning on synthetic scenes with compositional generalization"
4. Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## Step 2: Initialize and Push

In your terminal, from the `learning_to_see` directory:

```bash
# Initialize git (if not already initialized)
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit: Complete vision-language learning pipeline"

# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/learning_to_see.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 3: Update Notebook

After pushing to GitHub:

1. Edit `train_captioner.ipynb`
2. Find this line in cell 5:
   ```python
   GITHUB_REPO = "https://github.com/YOUR_USERNAME/learning_to_see.git"
   ```
3. Replace `YOUR_USERNAME` with your actual GitHub username
4. Save and commit:
   ```bash
   git add train_captioner.ipynb
   git commit -m "Update notebook with correct GitHub URL"
   git push
   ```

## Step 4: Update README

1. Edit `README.md`
2. Replace all instances of `YOUR_USERNAME` with your GitHub username
3. Update the Colab badge URL
4. Commit and push:
   ```bash
   git add README.md
   git commit -m "Update README with correct URLs"
   git push
   ```

## Step 5: Use in Google Colab

Now you can use the notebook in Colab:

1. Go to https://colab.research.google.com
2. File → Open Notebook → GitHub tab
3. Enter your GitHub URL: `https://github.com/YOUR_USERNAME/learning_to_see`
4. Open `train_captioner.ipynb`
5. Run cells sequentially!

The first cell will automatically clone your repository and set up everything.

## Alternative: Direct Colab Link

Create a direct link by updating this URL:
```
https://colab.research.google.com/github/YOUR_USERNAME/learning_to_see/blob/main/train_captioner.ipynb
```

You can also add this badge to your README:
```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/learning_to_see/blob/main/train_captioner.ipynb)
```

## Troubleshooting

### Private Repository
If your repo is private, Colab will prompt you to authenticate with GitHub.

### Large Files
GitHub has a 100MB file size limit. This project doesn't include large files, but:
- Data is generated in Colab (not stored in repo)
- Model checkpoints are saved to Google Drive (not in repo)

### Updates
To update the code after making changes:
```bash
git add .
git commit -m "Description of changes"
git push
```

Then in Colab, restart runtime and re-run the first cell to get the latest code.

## Quick Commands Reference

```bash
# Check status
git status

# See changes
git diff

# Add specific files
git add filename.py

# Commit with message
git commit -m "Your message"

# Push changes
git push

# Pull latest changes
git pull

# Create new branch
git checkout -b feature-name

# Switch branches
git checkout main
```
