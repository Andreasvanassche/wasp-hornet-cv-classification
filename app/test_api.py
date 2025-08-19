#!/usr/bin/env python3
"""
Test script for the Hornet & Wasp Classification API
"""

import requests
import json
from pathlib import Path
import sys

def test_health_endpoint():
    """Test the health check endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_classes_endpoint():
    """Test the classes endpoint"""
    print("\n📋 Testing classes endpoint...")
    try:
        response = requests.get("http://localhost:8000/classes")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Classes endpoint: {data}")
            return True
        else:
            print(f"❌ Classes endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Classes endpoint error: {e}")
        return False

def test_prediction_endpoint(image_path=None):
    """Test the prediction endpoint"""
    print("\n🎯 Testing prediction endpoint...")
    
    if not image_path:
        # Look for test images in the dataset
        possible_paths = [
            "../dataset/data3000/data/val/images/Vespa_crabro",
            "../dataset/data3000/data/val/images/Vespa_velutina", 
            "../dataset/data3000/data/val/images/Vespula_sp"
        ]
        
        test_image = None
        for path_str in possible_paths:
            path = Path(path_str)
            if path.exists():
                jpg_files = list(path.glob("*.jpg"))
                if jpg_files:
                    test_image = jpg_files[0]
                    break
        
        if not test_image:
            print("⚠️  No test image found. Skipping prediction test.")
            return True
        
        image_path = test_image
    
    try:
        print(f"📷 Using test image: {image_path}")
        
        with open(image_path, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            response = requests.post("http://localhost:8000/predict", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Prediction successful!")
            print(f"   Predicted class: {data['predicted_class']}")
            print(f"   Confidence: {data['confidence']:.3f}")
            print(f"   Processing time: {data['processing_time']:.3f}s")
            print("   All probabilities:")
            for species, prob in data['probabilities'].items():
                print(f"     {species}: {prob:.3f}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def main():
    """Run all API tests"""
    print("🧪 Hornet & Wasp Classifier API Tests")
    print("=" * 40)
    
    # Check if server is running
    print("🔌 Checking if server is running...")
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code != 200:
            print("❌ Server not responding. Is the app running?")
            print("   Try: ./run.sh or docker-compose up")
            sys.exit(1)
        print("✅ Server is running")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("   Try: ./run.sh or docker-compose up")
        sys.exit(1)
    
    # Run tests
    tests = [
        test_health_endpoint,
        test_classes_endpoint,
        test_prediction_endpoint
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Summary")
    print("=" * 40)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All tests passed! ({passed}/{total})")
        print("\n🎉 Your API is working correctly!")
        print("\n🌐 You can now:")
        print("   • Visit: http://localhost:8000")
        print("   • API Docs: http://localhost:8000/docs")
        print("   • Upload images and get predictions!")
    else:
        print(f"❌ Some tests failed ({passed}/{total})")
        print("\n🔧 Check the application logs:")
        print("   docker-compose logs -f")
        sys.exit(1)

if __name__ == "__main__":
    main()