#!/usr/bin/env python3
"""
Download model from storage
Usage:
    python download_model.py --version v1.0.0 --storage s3
    python download_model.py --version production --output ../backend/saved_model
"""
import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.model_storage import get_storage


def main():
    parser = argparse.ArgumentParser(description='Download model from storage')
    parser.add_argument('--version', required=True, help='Model version to download')
    parser.add_argument('--output', default='./saved_model', help='Output directory')
    parser.add_argument('--storage', choices=['s3', 'huggingface', 'local'],
                       default='local', help='Storage backend')
    parser.add_argument('--force', action='store_true', help='Overwrite if exists')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    # Check if already exists
    if output_dir.exists() and not args.force:
        print(f"❌ Error: Output directory already exists: {output_dir}")
        print("Use --force to overwrite")
        sys.exit(1)
    
    # Download
    try:
        storage = get_storage(args.storage)
        print(f"Downloading {args.version} from {args.storage}")
        storage.download_model(args.version, str(output_dir))
        
        print(f"\n✅ Success! Model {args.version} downloaded to {output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
