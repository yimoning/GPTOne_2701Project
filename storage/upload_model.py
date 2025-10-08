#!/usr/bin/env python3
"""
Upload trained model to storage
Usage:
    python upload_model.py --version v1.0.0 --storage s3
    python upload_model.py --version v1.0.0 --storage huggingface
    python upload_model.py --version v1.0.0 --storage local
"""
import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.model_storage import get_storage


def main():
    parser = argparse.ArgumentParser(description='Upload model to storage')
    parser.add_argument('--version', required=True, help='Model version to upload')
    parser.add_argument('--model-dir', default=None, help='Local model directory (default: ../training/models/{version})')
    parser.add_argument('--storage', choices=['s3', 'huggingface', 'local'], 
                       default='local', help='Storage backend')
    
    args = parser.parse_args()
    
    # Determine model directory
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        model_dir = Path(__file__).parent.parent / 'training' / 'models' / args.version
    
    if not model_dir.exists():
        print(f"❌ Error: Model directory not found: {model_dir}")
        print(f"\nAvailable models:")
        models_dir = Path(__file__).parent.parent / 'training' / 'models'
        if models_dir.exists():
            for d in sorted(models_dir.iterdir()):
                if d.is_dir():
                    print(f"  - {d.name}")
        sys.exit(1)
    
    # Upload
    try:
        storage = get_storage(args.storage)
        print(f"Uploading {args.version} from {model_dir}")
        storage.upload_model(str(model_dir), args.version)
        
        print(f"\n✅ Success! Model {args.version} uploaded to {args.storage}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
