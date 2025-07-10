import os
import time
from transformers import BertConfig, BertModel, AutoTokenizer

def download_bert_with_retry(max_retries=3, delay=2):
    """Download BERT model with retry logic and better error handling"""
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs("./checkpoints/bert-base-uncased", exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}: Downloading BERT model...")
            
            # Download with offline fallback and no authentication required
            config = BertConfig.from_pretrained(
                "bert-base-uncased",
                local_files_only=False,
                use_auth_token=False
            )
            
            model = BertModel.from_pretrained(
                "bert-base-uncased", 
                add_pooling_layer=False, 
                config=config,
                local_files_only=False,
                use_auth_token=False
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-uncased",
                local_files_only=False,
                use_auth_token=False
            )
            
            # Save locally
            config.save_pretrained("./checkpoints/bert-base-uncased")
            model.save_pretrained("./checkpoints/bert-base-uncased")
            tokenizer.save_pretrained("./checkpoints/bert-base-uncased")
            
            print("✓ Successfully downloaded and saved BERT model!")
            return True
            
        except Exception as e:
            print(f"✗ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print("All attempts failed. Trying alternative approach...")
                return False
    
    return False

def try_alternative_download():
    """Try alternative download methods"""
    try:
        print("Trying alternative download method...")
        
        # Try without any authentication headers
        os.environ.pop('HF_TOKEN', None)  # Remove any existing token
        
        config = BertConfig.from_pretrained("bert-base-uncased", trust_remote_code=True)
        model = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False, config=config, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", trust_remote_code=True)
        
        config.save_pretrained("./checkpoints/bert-base-uncased")
        model.save_pretrained("./checkpoints/bert-base-uncased")
        tokenizer.save_pretrained("./checkpoints/bert-base-uncased")
        
        print("✓ Alternative download method succeeded!")
        return True
        
    except Exception as e:
        print(f"✗ Alternative method also failed: {e}")
        return False

if __name__ == "__main__":
    # Clear any existing auth tokens that might be causing issues
    if 'HF_TOKEN' in os.environ:
        del os.environ['HF_TOKEN']
    
    success = download_bert_with_retry()
    
    if not success:
        success = try_alternative_download()
    
    if not success:
        print("\n❌ Could not download BERT model automatically.")
        print("Please try one of these solutions:")
        print("1. Check your internet connection")
        print("2. Try again later (Hugging Face might be experiencing issues)")
        print("3. Download manually from: https://huggingface.co/bert-base-uncased")
        print("4. Or use a different model that doesn't require authentication")
    else:
        print("\n🎉 BERT model downloaded successfully!")
        print("Model saved to: ./checkpoints/bert-base-uncased")
