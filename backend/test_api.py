"""
Simple test script to verify the API is working
"""
import requests
import json

API_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_prediction():
    """Test prediction endpoint with sample texts"""
    print("Testing /predict endpoint...")
    
    # Sample texts
    samples = [
        {
            "text": "Climate change is one of the most pressing issues facing humanity today. "
                   "The scientific consensus is clear: human activities are causing global temperatures to rise.",
            "expected": "Could be either - simple factual text"
        },
        {
            "text": "The quick brown fox jumps over the lazy dog.",
            "expected": "Too short for reliable prediction"
        }
    ]
    
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}: {sample['expected']}")
        print(f"Text: {sample['text'][:80]}...")
        
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": sample["text"]}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Result: {result['label']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Message: {result['message']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n" + "="*60)
    print("Testing /batch-predict endpoint...")
    
    texts = [
        "This is the first test essay.",
        "This is the second test essay.",
        "This is the third test essay."
    ]
    
    response = requests.post(
        f"{API_URL}/batch-predict",
        json=texts
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Processed {result['count']} texts")
        for i, res in enumerate(result['results'], 1):
            if 'error' not in res:
                print(f"\n{i}. {res['label']} ({res['confidence']:.2%} confidence)")
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    try:
        print("="*60)
        print("API Test Suite")
        print("="*60 + "\n")
        
        test_health()
        test_prediction()
        test_batch_prediction()
        
        print("\n" + "="*60)
        print("All tests completed!")
        print("Visit http://localhost:8000/docs for interactive API testing")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API")
        print("Make sure the API is running: uvicorn main:app --reload")
    except Exception as e:
        print(f"❌ Error: {e}")
