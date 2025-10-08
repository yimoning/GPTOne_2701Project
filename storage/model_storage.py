"""
Model storage abstraction - supports multiple backends (S3, HuggingFace, local)
"""
import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional


class ModelStorage(ABC):
    """Abstract base class for model storage"""
    
    @abstractmethod
    def upload_model(self, local_path: str, version: str):
        """Upload model to storage"""
        pass
    
    @abstractmethod
    def download_model(self, version: str, local_path: str):
        """Download model from storage"""
        pass
    
    @abstractmethod
    def list_versions(self) -> list:
        """List available model versions"""
        pass


class S3Storage(ModelStorage):
    """AWS S3 storage backend"""
    
    def __init__(self, bucket: Optional[str] = None):
        try:
            import boto3
        except ImportError:
            raise ImportError("Install boto3: pip install boto3")
        
        self.s3 = boto3.client('s3')
        self.bucket = bucket or os.getenv('MODEL_BUCKET', 'gptone-models')
    
    def upload_model(self, local_path: str, version: str):
        """Upload all files in model directory to S3"""
        print(f"Uploading model {version} to s3://{self.bucket}/")
        
        for file in Path(local_path).rglob('*'):
            if file.is_file():
                key = f'models/{version}/{file.relative_to(local_path)}'
                self.s3.upload_file(str(file), self.bucket, key)
                print(f'  ✓ {file.name}')
        
        print(f"✅ Model {version} uploaded successfully")
    
    def download_model(self, version: str, local_path: str):
        """Download model from S3"""
        print(f"Downloading model {version} from s3://{self.bucket}/")
        
        Path(local_path).mkdir(parents=True, exist_ok=True)
        
        # List all files for this version
        paginator = self.s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket, Prefix=f'models/{version}/')
        
        for page in pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                filename = os.path.basename(key)
                local_file = os.path.join(local_path, filename)
                
                self.s3.download_file(self.bucket, key, local_file)
                print(f'  ✓ {filename}')
        
        print(f"✅ Model {version} downloaded successfully")
    
    def list_versions(self) -> list:
        """List available model versions in S3"""
        response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix='models/', Delimiter='/')
        
        versions = []
        for prefix in response.get('CommonPrefixes', []):
            version = prefix['Prefix'].split('/')[-2]
            versions.append(version)
        
        return sorted(versions)


class LocalStorage(ModelStorage):
    """Local or shared filesystem storage"""
    
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path or os.getenv('MODEL_PATH', './models'))
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def upload_model(self, local_path: str, version: str):
        """Copy to storage location"""
        import shutil
        
        dest = self.base_path / version
        if dest.exists():
            shutil.rmtree(dest)
        
        shutil.copytree(local_path, dest)
        print(f'✅ Model {version} copied to {dest}')
    
    def download_model(self, version: str, local_path: str):
        """Copy from storage location"""
        import shutil
        
        source = self.base_path / version
        if not source.exists():
            raise FileNotFoundError(f"Model version {version} not found at {source}")
        
        Path(local_path).mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, local_path, dirs_exist_ok=True)
        print(f'✅ Model {version} copied from {source}')
    
    def list_versions(self) -> list:
        """List available model versions"""
        versions = [d.name for d in self.base_path.iterdir() if d.is_dir()]
        return sorted(versions)


class HuggingFaceStorage(ModelStorage):
    """HuggingFace Hub storage"""
    
    def __init__(self, repo_id: Optional[str] = None):
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError("Install huggingface_hub: pip install huggingface_hub")
        
        self.api = HfApi()
        self.repo_id = repo_id or os.getenv('HF_REPO_ID', 'username/gptone-detector')
    
    def upload_model(self, local_path: str, version: str):
        """Upload to HuggingFace Hub"""
        print(f"Uploading model {version} to HuggingFace: {self.repo_id}")
        
        self.api.upload_folder(
            folder_path=local_path,
            repo_id=self.repo_id,
            revision=version,
            repo_type="model"
        )
        
        print(f'✅ Model uploaded to https://huggingface.co/{self.repo_id}/tree/{version}')
    
    def download_model(self, version: str, local_path: str):
        """Download from HuggingFace Hub"""
        from transformers import AutoModel
        
        print(f"Downloading from HuggingFace: {self.repo_id}@{version}")
        
        # HuggingFace transformers handles download automatically
        model = AutoModel.from_pretrained(self.repo_id, revision=version)
        model.save_pretrained(local_path)
        
        print(f'✅ Model downloaded to {local_path}')
    
    def list_versions(self) -> list:
        """List available versions (branches/tags)"""
        refs = self.api.list_repo_refs(self.repo_id)
        versions = [ref.name for ref in refs.branches] + [ref.name for ref in refs.tags]
        return sorted(versions)


def get_storage(storage_type: Optional[str] = None) -> ModelStorage:
    """
    Factory function to get storage backend
    
    Args:
        storage_type: 's3', 'huggingface', or 'local' (default: from env or 'local')
    
    Returns:
        ModelStorage instance
    
    Environment variables:
        MODEL_STORAGE: Storage type (s3, huggingface, local)
        MODEL_BUCKET: S3 bucket name (for s3)
        MODEL_PATH: Base path (for local)
        HF_REPO_ID: HuggingFace repo (for huggingface)
    
    Examples:
        >>> # Use S3
        >>> storage = get_storage('s3')
        >>> storage.upload_model('./models/v1.0.0', 'v1.0.0')
        
        >>> # Use local storage
        >>> storage = get_storage('local')
        >>> storage.download_model('v1.0.0', './saved_model')
    """
    storage_type = storage_type or os.getenv('MODEL_STORAGE', 'local')
    
    if storage_type == 's3':
        return S3Storage()
    elif storage_type == 'huggingface':
        return HuggingFaceStorage()
    elif storage_type == 'local':
        return LocalStorage()
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")


if __name__ == "__main__":
    # CLI for testing
    import argparse
    
    parser = argparse.ArgumentParser(description='Model storage utility')
    parser.add_argument('action', choices=['upload', 'download', 'list'])
    parser.add_argument('--version', help='Model version')
    parser.add_argument('--local-path', help='Local path', default='./saved_model')
    parser.add_argument('--storage', choices=['s3', 'huggingface', 'local'], default='local')
    
    args = parser.parse_args()
    
    storage = get_storage(args.storage)
    
    if args.action == 'upload':
        if not args.version:
            parser.error("--version required for upload")
        storage.upload_model(args.local_path, args.version)
    
    elif args.action == 'download':
        if not args.version:
            parser.error("--version required for download")
        storage.download_model(args.version, args.local_path)
    
    elif args.action == 'list':
        versions = storage.list_versions()
        print(f"Available versions ({len(versions)}):")
        for v in versions:
            print(f"  - {v}")
