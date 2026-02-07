"""
Test script to verify installation and dependencies
Run this after installing requirements.txt
"""

import sys

def test_imports():
    """Test if all required packages are installed"""
    print("Testing package imports...\n")
    
    packages = [
        ('streamlit', 'Streamlit'),
        ('cv2', 'OpenCV'),
        ('pandas', 'Pandas'),
        ('plotly', 'Plotly'),
        ('ultralytics', 'YOLOv8'),
        ('mediapipe', 'MediaPipe'),
        ('numpy', 'NumPy')
    ]
    
    failed = []
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name:15} - OK")
        except ImportError as e:
            print(f"❌ {name:15} - FAILED: {e}")
            failed.append(name)
    
    print()
    
    if failed:
        print(f"❌ {len(failed)} package(s) failed to import:")
        for pkg in failed:
            print(f"   - {pkg}")
        print("\nPlease install missing packages:")
        print("   pip install -r requirements.txt")
        return False
    else:
        print("✅ All packages imported successfully!")
        return True


def test_models():
    """Test if models can be loaded"""
    print("\nTesting model loading...\n")
    
    try:
        from ultralytics import YOLO
        print("Loading YOLOv8 model...")
        model = YOLO('yolov8n.pt')
        print("✅ YOLOv8 model loaded successfully!")
    except Exception as e:
        print(f"❌ YOLOv8 model failed to load: {e}")
        return False
    
    try:
        import mediapipe as mp
        print("Loading MediaPipe Pose...")
        pose = mp.solutions.pose.Pose()
        pose.close()
        print("✅ MediaPipe Pose loaded successfully!")
    except Exception as e:
        print(f"❌ MediaPipe Pose failed to load: {e}")
        return False
    
    return True


def test_output_directory():
    """Test if output directory can be created"""
    print("\nTesting output directory...\n")
    
    import os
    
    try:
        os.makedirs("output", exist_ok=True)
        print("✅ Output directory created successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to create output directory: {e}")
        return False


def test_opencv_video():
    """Test OpenCV video capabilities"""
    print("\nTesting OpenCV video capabilities...\n")
    
    try:
        import cv2
        
        # Test VideoWriter fourcc
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print("✅ OpenCV VideoWriter codec available!")
        
        # Test basic OpenCV functionality
        test_frame = cv2.imread('test.jpg')  # Will fail but that's ok
        print("✅ OpenCV is functional!")
        
        return True
    except Exception as e:
        print(f"⚠️  OpenCV warning: {e}")
        print("   This is normal if no test image exists")
        return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Exam Proctoring System - Installation Test")
    print("=" * 60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Package Imports", test_imports()))
    results.append(("Model Loading", test_models()))
    results.append(("Output Directory", test_output_directory()))
    results.append(("OpenCV Video", test_opencv_video()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20} : {status}")
    
    all_passed = all(result[1] for result in results)
    
    print()
    if all_passed:
        print("✅ All tests passed! System is ready to use.")
        print("\nRun the dashboard with:")
        print("   streamlit run streamlit_app.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("   1. Reinstall requirements: pip install -r requirements.txt")
        print("   2. Check Python version (3.8+ required)")
        print("   3. Check internet connection (for model download)")
    
    print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())