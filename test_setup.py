#!/usr/bin/env python3
"""
Simple test to verify environment setup
"""

def test_imports():
    """Test all required imports"""
    print("Testing environment setup...")
    print("=" * 40)
    
    imports = [
        "os",
        "numpy", 
        "matplotlib.pyplot",
        "sklearn.model_selection",
        "cv2",
        "tqdm"
    ]
    
    success = 0
    for module in imports:
        try:
            exec(f"import {module}")
            print(f"✅ {module}")
            success += 1
        except ImportError as e:
            print(f"❌ {module}: {e}")
    
    # Test Keras (try both standalone and TensorFlow)
    keras_success = False
    try:
        from keras.preprocessing.image import ImageDataGenerator
        print("✅ keras (standalone)")
        keras_success = True
    except ImportError:
        try:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            print("✅ tensorflow.keras")
            keras_success = True
        except ImportError:
            print("❌ keras/tensorflow.keras")
    
    if keras_success:
        success += 1
    
    print("=" * 40)
    print(f"Result: {success}/{len(imports)+1} modules working")
    
    if success == len(imports) + 1:
        print("🎉 Environment setup successful!")
        return True
    else:
        print("❗ Some modules missing")
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\nNext step: Run data_preprocessing.py")
    else:
        print("\nPlease install missing modules")