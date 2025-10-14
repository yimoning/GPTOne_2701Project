"""
Production training pipeline for AI text detection model
Refactored from notebook for reproducibility and automation
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import argparse
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training, evaluation, and versioning"""
    
    def __init__(self, config):
        self.config = config
        self.seq_length = config['seq_length']
        self.device = self._setup_device()
        self.model = None
        self.tokenizer = None
        
    def _setup_device(self):
        """Determine best available device"""
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        logger.info(f'Using device: {device}')
        return device
    
    def load_data(self, data_path):
        """Load and prepare training data"""
        logger.info(f"Loading data from {data_path}")
        
        # Load datasets
        df_train_essays = pd.read_csv(f"{data_path}/train_essays.csv")
        df_train_essays_ext = pd.read_csv(f"{data_path}/train_v2_drcat_02.csv")
        df_train_essays_ext.rename(columns={"label": "generated"}, inplace=True)
        
        # Combine datasets
        df_combined = pd.concat([
            df_train_essays_ext[["text", "generated"]], 
            df_train_essays[["text", "generated"]]
        ])
        
        logger.info(f"Total samples: {len(df_combined)}")
        logger.info(f"AI-generated: {df_combined['generated'].sum()}")
        logger.info(f"Human-written: {(df_combined['generated'] == 0).sum()}")
        
        return df_combined
    
    def prepare_datasets(self, df, test_size=0.33, random_state=42):
        """Split and tokenize datasets"""
        logger.info("Preparing datasets...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"], 
            df["generated"],
            test_size=test_size,
            random_state=random_state
        )
        
        # Create HuggingFace datasets
        train_dataset = Dataset.from_dict({
            'text': X_train.tolist(), 
            'label': y_train.tolist()
        })
        test_dataset = Dataset.from_dict({
            'text': X_test.tolist(), 
            'label': y_test.tolist()
        })
        
        # Tokenize
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'], 
                truncation=True, 
                padding='max_length', 
                max_length=self.seq_length
            )
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        
        return train_dataset, test_dataset, y_test
    
    def initialize_model(self):
        """Initialize model architecture"""
        logger.info("Initializing model...")
        
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=2
        )
        
        # Freeze backbone if specified
        if self.config.get('freeze_backbone', True):
            for param in self.model.distilbert.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen - only training classification head")
        
        self.model.to(self.device)
        
    def train(self, train_dataset, test_dataset, y_test):
        """Train the model"""
        logger.info("Starting training...")
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        # Training metrics
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_acc': [],
            'epochs': []
        }
        
        for epoch in range(self.config['epochs']):
            # Training phase
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(output.logits, labels)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                correct += torch.sum(torch.argmax(output.logits, dim=1) == labels).item()
                total += len(input_ids)
            
            train_loss = total_loss / total
            train_acc = correct / total
            
            # Validation phase
            val_acc = self._evaluate(test_dataloader)
            
            # Log metrics
            logger.info(
                f"Epoch {epoch+1} - "
                f"Loss: {train_loss:.4f}, "
                f"Train Acc: {train_acc:.4f}, "
                f"Val Acc: {val_acc:.4f}"
            )
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['epochs'].append(epoch + 1)
        
        # Final evaluation
        logger.info("\nFinal Evaluation:")
        final_metrics = self._detailed_evaluation(test_dataloader, y_test)
        
        return history, final_metrics
    
    def _evaluate(self, dataloader):
        """Quick accuracy evaluation"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pred = torch.argmax(output.logits, dim=1)
                
                correct += (pred == labels).sum().item()
                total += len(labels)
        
        return correct / total
    
    def _detailed_evaluation(self, dataloader, y_true):
        """Detailed evaluation with metrics"""
        self.model.eval()
        y_pred = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pred = torch.argmax(output.logits, dim=1)
                y_pred.extend(pred.cpu().tolist())
        
        # Calculate metrics
        accuracy = (y_true.to_numpy() == y_pred).mean()
        conf_matrix = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=['Human', 'AI'], output_dict=True)
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"\nConfusion Matrix:\n{conf_matrix}")
        logger.info(f"\nClassification Report:\n{classification_report(y_true, y_pred, target_names=['Human', 'AI'])}")
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': report
        }
    
    def save_model(self, output_dir, version, metrics):
        """Save model with versioning"""
        # Create version directory
        version_dir = Path(output_dir) / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(version_dir)
        self.tokenizer.save_pretrained(version_dir)
        
        # Save metadata
        metadata = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'metrics': metrics,
            'device': str(self.device)
        }
        
        with open(version_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {version_dir}")
        
        # Create/update 'latest' symlink (Unix only)
        try:
            latest_link = Path(output_dir) / 'latest'
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(version_dir.name)
        except Exception as e:
            logger.warning(f"Could not create symlink: {e}")
        
        return version_dir


def main():
    parser = argparse.ArgumentParser(description='Train AI text detection model')
    parser.add_argument('--data-path', type=str, default='../data', help='Path to training data')
    parser.add_argument('--output-dir', type=str, default='./models', help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--seq-length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--version', type=str, default=None, help='Model version (default: timestamp)')
    parser.add_argument('--freeze-backbone', action='store_true', default=True, help='Freeze BERT backbone')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'seq_length': args.seq_length,
        'freeze_backbone': args.freeze_backbone
    }
    
    # Version
    version = args.version or datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Train
    trainer = ModelTrainer(config)
    df = trainer.load_data(args.data_path)
    train_dataset, test_dataset, y_test = trainer.prepare_datasets(df)
    trainer.initialize_model()
    history, metrics = trainer.train(train_dataset, test_dataset, y_test)
    
    # Save
    model_path = trainer.save_model(args.output_dir, version, metrics)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete!")
    logger.info(f"Model version: {version}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
