"""
Helper script to export your trained model from the notebook
Run this after training your model in the notebook
"""
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from pathlib import Path

def save_model_for_api(model, tokenizer, output_dir='./backend/saved_model'):
    """
    Save trained model and tokenizer for the API
    
    Args:
        model: Your trained DistilBertForSequenceClassification model
        tokenizer: Your DistilBertTokenizer
        output_dir: Where to save (default: ./backend/saved_model)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving model to {output_path}...")
    
    # Save model
    model.save_pretrained(output_path)
    print("✓ Model saved")
    
    # Save tokenizer
    tokenizer.save_pretrained(output_path)
    print("✓ Tokenizer saved")
    
    # Verify files
    expected_files = ['config.json', 'tokenizer_config.json', 'vocab.txt']
    model_file = output_path / 'model.safetensors'
    if not model_file.exists():
        model_file = output_path / 'pytorch_model.bin'
    
    if model_file.exists():
        print(f"✓ Model weights saved: {model_file.name}")
    else:
        print("⚠ Warning: Model weights file not found")
    
    for file in expected_files:
        if (output_path / file).exists():
            print(f"✓ {file}")
        else:
            print(f"⚠ Warning: {file} not found")
    
    print("\n✅ Model export complete!")
    print(f"📁 Location: {output_path.absolute()}")
    print("\nNext steps:")
    print("1. cd backend")
    print("2. pip install -r requirements.txt")
    print("3. uvicorn main:app --reload")

# Example usage (add this to your notebook):
"""
# After training your model in the notebook:

from save_model import save_model_for_api

# Save the model
save_model_for_api(model, tokenizer)

# Or specify a different location:
# save_model_for_api(model, tokenizer, './my_model_directory')
"""

if __name__ == "__main__":
    print("This is a helper module to save your model.")
    print("Import and use save_model_for_api() in your notebook after training.")
    print("\nExample:")
    print("  from save_model import save_model_for_api")
    print("  save_model_for_api(model, tokenizer)")
