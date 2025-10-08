# Training Pipeline

Production training for AI text detection models with versioning.

## Requirements

- Python 3.8+
- Dataset: `../data/train_essays.csv` with `text` and `generated` columns

## Setup

```bash
cd training
python -m venv venv_training
source venv_training/bin/activate
pip install -r requirements.txt
```

## Training

```bash
# Basic
python train.py --version v1.0.0

# Custom configuration
python train.py --version v1.1.0 --epochs 5 --batch-size 16 --learning-rate 2e-5

# All options
python train.py --help
```

**Key Parameters:**
- `--version`: Model version (required)
- `--epochs`: Training epochs (default: 3)
- `--batch-size`: Batch size (default: 8)
- `--learning-rate`: Learning rate (default: 2e-5)
- `--data-path`: Training data path
- `--output-dir`: Output directory (default: ./models)

## Output

Models saved to `./models/{version}/`:
```
models/v1.0.0/
├── pytorch_model.bin
├── config.json
├── tokenizer files
└── metadata.json        # Training params, metrics, timestamp
```

## Versioning

Use semantic versioning: `vMAJOR.MINOR.PATCH`
- `v1.0.0`: Initial model
- `v1.1.0`: New features/data
- `v2.0.0`: Breaking changes

Mark production version:
```bash
cd models && ln -sf v1.0.0 production
```

## Automated Retraining

```bash
./retrain.sh              # Auto-increments version
./retrain.sh v1.2.0 5     # Custom version and epochs
```

Script handles: version increment, training, evaluation, production promotion.
