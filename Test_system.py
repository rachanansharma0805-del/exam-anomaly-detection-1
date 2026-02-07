"""
Test Script for Exam Proctoring System
Run this to validate system functionality
"""

import cv2
import numpy as np
import os
import sys
from datetime import datetime

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking dependencies...")
    
    dependencies = {
        'cv2': 'OpenCV',
        'mediapipe': 'MediaPipe',
        'streamlit': 'Streamlit',
        'pandas': 'Pandas',
        'plotly': 'Plotly',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'scipy': 'SciPy'
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} missing")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed\n")
    return True


def check_modules():
    """Check if all custom modules can be imported"""
    print("Checking custom modules...")
    
    modules = [
        'person_detector',
        'pose_estimator',
        'anomaly_detector',
        'video_processor'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}.py found and importable")
        except ImportError as e:
            print(f"✗ {module}.py error: {e}")
            return False
    
    print("\n✅ All modules importable\n")
    return True


def create_test_video():
    """Create a simple test video for validation"""
    print("Creating test video...")
    
    output_path = "test_video.mp4"
    
    # Video properties
    width, height = 640, 480
    fps = 30
    duration = 5  # seconds
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = fps * duration
    
    for frame_num in range(total_frames):
        # Create blank frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 240
        
        # Add text
        text = f"Test Frame {frame_num}/{total_frames}"
        cv2.putText(frame, text, (50, height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Simulate a moving "person" (rectangle)
        x = int((width - 200) * (frame_num / total_frames))
        y = height // 3
        cv2.rectangle(frame, (x, y), (x + 100, y + 200), (100, 150, 200), -1)
        
        out.write(frame)
    
    out.release()
    
    if os.path.exists(output_path):
        print(f"✓ Test video created: {output_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Duration: {duration}s")
        print(f"  FPS: {fps}\n")
        return output_path
    else:
        print("✗ Failed to create test video\n")
        return None


def test_processing_pipeline(video_path):
    """Test the video processing pipeline"""
    print("Testing processing pipeline...")
    
    try:
        from video_processor import ExamProctorPipeline
        
        # Initialize pipeline
        pipeline = ExamProctorPipeline(output_dir="test_output")
        print("✓ Pipeline initialized")
        
        # Process video (just a few frames for testing)
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            print("✓ Video readable")
            print(f"  Frame shape: {frame.shape}")
        else:
            print("✗ Could not read video")
            return False
        
        # Test person detector
        from person_detector import PersonDetector
        detector = PersonDetector()
        detections = detector.detect_persons(frame)
        print(f"✓ Person detector working (detected {len(detections)} persons)")
        
        # Test pose estimator
        from pose_estimator import PoseEstimator
        estimator = PoseEstimator()
        print("✓ Pose estimator initialized")
        estimator.cleanup()
        
        # Test anomaly detector
        from anomaly_detector import AnomalyDetector
        estimator2 = PoseEstimator()
        anomaly_det = AnomalyDetector(estimator2)
        print("✓ Anomaly detector initialized")
        estimator2.cleanup()
        
        print("\n✅ All pipeline components working\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Pipeline test failed: {e}\n")
        return False


def test_streamlit_app():
    """Check if Streamlit app can be imported"""
    print("Testing Streamlit app...")
    
    try:
        # Check if file exists
        if not os.path.exists('streamlit_app.py'):
            print("✗ streamlit_app.py not found")
            return False
        
        print("✓ streamlit_app.py found")
        print("  To run: streamlit run streamlit_app.py\n")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}\n")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("EXAM PROCTORING SYSTEM - VALIDATION TEST")
    print("="*60)
    print()
    
    # Run tests
    tests = [
        ("Dependencies", check_dependencies),
        ("Custom Modules", check_modules),
        ("Streamlit App", test_streamlit_app),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"--- {test_name} ---")
        results[test_name] = test_func()
        print()
    
    # Create and test with test video
    print("--- Test Video ---")
    test_video = create_test_video()
    if test_video:
        results["Test Video"] = True
        print("--- Processing Pipeline ---")
        results["Pipeline"] = test_processing_pipeline(test_video)
    else:
        results["Test Video"] = False
        results["Pipeline"] = False
    
    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run: streamlit run streamlit_app.py")
        print("2. Upload an exam video")
        print("3. Click 'Process Video'")
        print("4. Review results\n")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)