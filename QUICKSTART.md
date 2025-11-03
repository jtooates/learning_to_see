# Quick Start Guide

Get started with the learning_to_see project in 3 steps!

## 🚀 Option 1: Google Colab (Recommended)

**Best for**: Quick experimentation with GPU access

1. **Push to GitHub** (follow [GITHUB_SETUP.md](GITHUB_SETUP.md))

2. **Open in Colab**:
   - Go to https://colab.research.google.com
   - File → Open Notebook → GitHub tab
   - Enter: `jtooates/learning_to_see`
   - Open `train_captioner.ipynb`

3. **Run the notebook**:
   - First cell clones the repo automatically
   - Run cells sequentially
   - Training takes ~30-45 min on A100 GPU

That's it! The notebook handles everything:
- ✅ Data generation
- ✅ Model training
- ✅ Visualization
- ✅ Evaluation
- ✅ Error analysis

## 💻 Option 2: Local Setup

**Best for**: Development and experimentation

```bash
# Clone repository
git clone https://github.com/jtooates/learning_to_see.git
cd learning_to_see

# Install dependencies
pip install -r requirements.txt

# Generate data
python -m data.gen \
    --output_dir ./data/scenes \
    --num_train 5000 \
    --num_val 500 \
    --num_test 500 \
    --split random \
    --seed 42

# Run tests
pytest tests/ -v

# Start training (or use the notebook)
jupyter notebook train_captioner.ipynb
```

## 🔬 What You'll See

### 1. Data Visualization
- Synthetic scenes with spatial relations
- Augmentation effects
- Distribution analysis

### 2. Training Progress
- Real-time loss curves
- Exact match accuracy
- Per-attribute F1 scores
- Learning rate schedule

### 3. Model Analysis
- Prediction visualizations (correct vs incorrect)
- Attention heatmaps
- Error analysis
- Confusion matrices

### 4. Compositional Generalization
- Performance on IID test set
- Color-shape holdout evaluation
- Relation holdout evaluation
- Comparison charts

## 📊 Expected Results

On 5K training samples (50 epochs):

| Metric | IID | Color Holdout | Relation Holdout |
|--------|-----|---------------|------------------|
| Exact Match | ~95% | ~85% | ~90% |
| Token Accuracy | ~98% | ~92% | ~95% |
| Color F1 | ~99% | ~90% | ~98% |
| Shape F1 | ~99% | ~88% | ~97% |

## 🎯 Next Steps

### Experiment with Hyperparameters
Edit these in the notebook:
- `batch_size`: 128 (for A100), 64 (for T4)
- `lr`: 3e-4 (default), try 1e-4 or 5e-4
- `max_epochs`: 50 (default), try 100 for better convergence
- `dropout`: 0.3 (default), try 0.1-0.5

### Try Different Splits
```bash
# Compositional splits
python -m data.gen \
    --output_dir ./data/scenes_comp \
    --num_train 5000 \
    --split color_shape \
    --seed 42
```

Available splits:
- `random`: Standard IID split
- `color_shape`: Holdout specific color-shape combinations
- `count_shape`: Holdout specific count-shape combinations
- `relation`: Holdout specific spatial relations

### Visualize Samples
```bash
python visualize_samples.py --data_dir ./data/scenes --num_samples 16
```

### Run Specific Tests
```bash
# DSL tests
pytest tests/test_dsl.py -v

# Renderer tests
pytest tests/test_renderer.py -v

# Captioner tests
pytest tests/test_captioner.py -v
```

## 📁 Project Structure

```
learning_to_see/
├── train_captioner.ipynb    ← Start here! Interactive training
├── dsl/                      ← Scene language (32 tokens)
├── render/                   ← Image generation (64×64 RGB)
├── data/                     ← Dataset pipeline
├── captioner/                ← ConvNeXt + GRU model
└── tests/                    ← 57 unit tests
```

## ❓ Common Issues

### Out of Memory
- Reduce `batch_size` in notebook
- Use smaller model (reduce `hidden_dim`)
- Reduce training data size

### Slow Training
- Use A100 GPU in Colab (Runtime → Change runtime type)
- Enable AMP (already enabled by default)
- Reduce `num_train` samples for faster iteration

### Import Errors
- Make sure you're in the project directory
- Check Python path is set correctly
- Verify all dependencies installed: `pip install -r requirements.txt`

### GitHub Clone Fails
- Update `GITHUB_REPO` URL in notebook cell 5
- Make sure repository is public (or authenticate for private)
- Check GitHub username is correct

## 📚 Learn More

- **Architecture**: See model diagrams in README.md
- **Grammar**: See dsl/ebnf.txt for full DSL specification
- **Examples**: Check examples.png and examples_detailed.png
- **Tests**: All tests have detailed docstrings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Submit a pull request

## 📝 Citation

```bibtex
@misc{learning_to_see,
  title={Learning to See: Vision-Language Learning on Synthetic Scenes},
  author={J. T. Oates},
  year={2024},
  url={https://github.com/jtooates/learning_to_see}
}
```

---

**Happy experimenting! 🎉**

For detailed GitHub setup instructions, see [GITHUB_SETUP.md](GITHUB_SETUP.md).
