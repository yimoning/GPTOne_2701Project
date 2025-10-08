# Model Storage

Store and retrieve models from local, S3, or HuggingFace Hub.

## Backends

- **Local**: No setup, fast, development/testing
- **S3**: Scalable, production, team sharing (~$1-2/month)
- **HuggingFace**: Free public repos, model sharing

## Requirements

```bash
pip install -r requirements.txt
```

For specific backends:
- S3: `boto3`
- HuggingFace: `huggingface_hub`

## Setup

**Local (default)**
```bash
export MODEL_STORAGE=local
```

**S3**
```bash
aws configure
export MODEL_STORAGE=s3
export MODEL_BUCKET=my-bucket
```

**HuggingFace**
```bash
huggingface-cli login
export MODEL_STORAGE=huggingface
export HF_REPO_ID=username/repo
```

## Usage

```bash
# Upload model
python upload_model.py --version v1.0.0 --storage s3

# Download model
python download_model.py --version v1.0.0 --storage s3

# List models
python model_storage.py list --storage s3
```

## Python API

```python
from model_storage import get_storage

storage = get_storage("s3")
storage.save_model("v1.0.0", "../training/models/v1.0.0")
storage.load_model("v1.0.0", "../backend/saved_model")
versions = storage.list_versions()
```

## Environment Variables

```bash
MODEL_STORAGE=local|s3|huggingface
MODEL_BUCKET=my-bucket          # S3
HF_REPO_ID=username/repo        # HuggingFace
AWS_ACCESS_KEY_ID=xxx           # S3
AWS_SECRET_ACCESS_KEY=yyy       # S3
HF_TOKEN=hf_xxx                 # HuggingFace
```
