"""
Example Usage Scripts for Exam Proctoring System
Complete examples for different use cases
"""

# ============================================================================
# EXAMPLE 1: Basic Video Processing
# ============================================================================

def example_basic_processing():
    """Basic video processing with default settings"""
    from video_processor import ExamProctorPipeline
    
    # Initialize pipeline
    pipeline = ExamProctorPipeline(output_dir="output")
    
    # Process video
    print("Processing exam video...")
    output_video, csv_path, summary = pipeline.process_video(
        video_path="exam_hall_recording.mp4"
    )
    
    # Print results
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Output Video: {output_video}")
    print(f"CSV Log: {csv_path}")
    print(f"\nTotal Anomalies: {summary['total_anomalies']}")
    print(f"Students Detected: {summary['total_students']}")
    print(f"Invigilators Detected: {summary['total_invigilators']}")
    
    if summary['total_anomalies'] > 0:
        print("\nAnomaly Breakdown:")
        for anomaly_type, count in summary['anomaly_types'].items():
            print(f"  - {anomaly_type}: {count}")
    
    # Cleanup
    pipeline.cleanup()


# ============================================================================
# EXAMPLE 2: Custom Configuration
# ============================================================================

def example_custom_config():
    """Processing with custom detection parameters"""
    from video_processor import ExamProctorPipeline
    
    # Initialize with custom output directory
    pipeline = ExamProctorPipeline(output_dir="custom_output")
    
    # Customize detection parameters
    pipeline.person_detector.confidence_threshold = 0.5  # More strict
    pipeline.anomaly_detector.head_turn_threshold = 40   # Less sensitive
    pipeline.anomaly_detector.hand_raise_threshold = 120 # Less sensitive
    
    # Process video
    output_video, csv_path, summary = pipeline.process_video(
        video_path="exam.mp4",
        output_video_path="custom_output/my_annotated_video.mp4",
        csv_path="custom_output/my_log.csv"
    )
    
    print(f"Results saved to custom_output/")
    pipeline.cleanup()


# ============================================================================
# EXAMPLE 3: Batch Processing Multiple Videos
# ============================================================================

def example_batch_processing():
    """Process multiple videos in batch"""
    import os
    from video_processor import ExamProctorPipeline
    import pandas as pd
    
    # List of videos to process
    video_files = [
        "exam_hall_1.mp4",
        "exam_hall_2.mp4",
        "exam_hall_3.mp4"
    ]
    
    # Initialize pipeline
    pipeline = ExamProctorPipeline(output_dir="batch_output")
    
    # Process each video
    all_summaries = []
    
    for idx, video_file in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing video {idx}/{len(video_files)}: {video_file}")
        print(f"{'='*60}")
        
        if not os.path.exists(video_file):
            print(f"Skipping {video_file} - file not found")
            continue
        
        try:
            # Process
            output_video, csv_path, summary = pipeline.process_video(
                video_path=video_file,
                output_video_path=f"batch_output/annotated_{idx}.mp4",
                csv_path=f"batch_output/log_{idx}.csv"
            )
            
            # Add to summaries
            summary['video_file'] = video_file
            all_summaries.append(summary)
            
            print(f"✓ Completed: {summary['total_anomalies']} anomalies detected")
            
        except Exception as e:
            print(f"✗ Error processing {video_file}: {e}")
    
    # Create summary report
    if all_summaries:
        df_summary = pd.DataFrame([
            {
                'Video': s['video_file'],
                'Total Anomalies': s['total_anomalies'],
                'Students': s['total_students'],
                'Invigilators': s['total_invigilators'],
                'Avg Severity': s.get('avg_severity', 0)
            }
            for s in all_summaries
        ])
        
        # Save summary
        df_summary.to_csv('batch_output/batch_summary.csv', index=False)
        
        print(f"\n{'='*60}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(df_summary.to_string(index=False))
        print(f"\nSummary saved to: batch_output/batch_summary.csv")
    
    pipeline.cleanup()


# ============================================================================
# EXAMPLE 4: Analyzing Results from CSV
# ============================================================================

def example_analyze_csv():
    """Analyze anomaly logs from CSV files"""
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Load CSV
    df = pd.read_csv("output/anomalies_20240101_120000.csv")
    
    print(f"Total anomalies in log: {len(df)}")
    
    # Group by person
    person_stats = df.groupby('person_id').agg({
        'anomaly_type': 'count',
        'severity': 'mean'
    }).rename(columns={'anomaly_type': 'count'})
    
    print("\nPer-Person Statistics:")
    print(person_stats)
    
    # Group by anomaly type
    type_counts = df['anomaly_type'].value_counts()
    print("\nAnomaly Type Distribution:")
    print(type_counts)
    
    # Find high-severity anomalies
    high_severity = df[df['severity'] > 70].sort_values('severity', ascending=False)
    print(f"\nHigh-Severity Anomalies ({len(high_severity)}):")
    print(high_severity[['timestamp', 'person_id', 'anomaly_type', 'severity', 'description']])
    
    # Timeline analysis
    df['minute'] = df['frame_number'] // (30 * 60)  # Assuming 30 FPS
    timeline = df.groupby('minute').size()
    
    print("\nAnomaly Timeline (by minute):")
    print(timeline)


# ============================================================================
# EXAMPLE 5: Real-time Monitoring (Frame by Frame)
# ============================================================================

def example_realtime_monitoring():
    """Process video frame-by-frame with custom actions"""
    import cv2
    from person_detector import PersonDetector
    from pose_estimator import PoseEstimator
    from anomaly_detector import AnomalyDetector
    
    # Initialize components
    detector = PersonDetector()
    estimator = PoseEstimator()
    anomaly_det = AnomalyDetector(estimator)
    
    # Open video
    cap = cv2.VideoCapture("exam.mp4")
    
    frame_count = 0
    anomaly_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect persons
        detections = detector.detect_persons(frame)
        tracked = detector.update_tracks(detections)
        
        # Process each person
        for person_id, detection in tracked.items():
            bbox = detection['bbox']
            
            # Estimate pose
            pose_data = estimator.estimate_pose(frame, bbox)
            
            if pose_data:
                # Detect anomalies
                anomalies = anomaly_det.detect_anomalies(
                    person_id, pose_data, bbox
                )
                
                if anomalies:
                    anomaly_count += len(anomalies)
                    
                    # Custom action: Alert on high-severity
                    for anomaly in anomalies:
                        if anomaly['severity'] > 80:
                            print(f"⚠️  HIGH SEVERITY at frame {frame_count}:")
                            print(f"   Person {person_id}: {anomaly['type']}")
                            print(f"   Severity: {anomaly['severity']:.1f}%")
        
        # Progress update
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames, {anomaly_count} anomalies")
    
    cap.release()
    estimator.cleanup()
    
    print(f"\nFinal: {frame_count} frames, {anomaly_count} total anomalies")


# ============================================================================
# EXAMPLE 6: Integration with External System
# ============================================================================

def example_external_integration():
    """Example of integrating with external reporting system"""
    from video_processor import ExamProctorPipeline
    import json
    import requests  # For API calls (example)
    
    # Process video
    pipeline = ExamProctorPipeline()
    output_video, csv_path, summary = pipeline.process_video("exam.mp4")
    
    # Prepare report for external system
    report = {
        'exam_id': 'EXAM_2024_001',
        'timestamp': '2024-01-01T12:00:00Z',
        'video_processed': output_video,
        'statistics': {
            'total_students': summary['total_students'],
            'total_anomalies': summary['total_anomalies'],
            'severity_average': summary.get('avg_severity', 0)
        },
        'violations': []
    }
    
    # Read CSV and add violations
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    for _, row in df[df['severity'] > 60].iterrows():
        report['violations'].append({
            'timestamp': row['timestamp'],
            'student_id': f"STUDENT_{row['person_id']:03d}",
            'violation_type': row['anomaly_type'],
            'severity': row['severity'],
            'requires_review': row['severity'] > 80
        })
    
    # Save JSON report
    with open('output/exam_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("Report generated: output/exam_report.json")
    
    # Example: Send to external API (uncomment if you have an API)
    # response = requests.post(
    #     'https://api.example.com/exam-reports',
    #     json=report,
    #     headers={'Authorization': 'Bearer YOUR_TOKEN'}
    # )
    # print(f"API Response: {response.status_code}")
    
    pipeline.cleanup()


# ============================================================================
# EXAMPLE 7: Generate Summary Report
# ============================================================================

def example_generate_report():
    """Generate a comprehensive text report"""
    import pandas as pd
    from datetime import datetime
    
    # Load results
    csv_path = "output/anomalies_20240101_120000.csv"
    df = pd.read_csv(csv_path)
    
    # Generate report
    report = f"""
{'='*70}
EXAM PROCTORING REPORT
{'='*70}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Video File: exam_hall_recording.mp4

{'='*70}
SUMMARY
{'='*70}
Total Anomalies Detected: {len(df)}
Unique Students Flagged: {df['person_id'].nunique()}
Average Severity: {df['severity'].mean():.1f}%

{'='*70}
ANOMALY BREAKDOWN
{'='*70}
"""
    
    # Add type counts
    for anom_type, count in df['anomaly_type'].value_counts().items():
        pct = (count / len(df)) * 100
        report += f"{anom_type:20s}: {count:3d} ({pct:5.1f}%)\n"
    
    report += f"""
{'='*70}
HIGH-SEVERITY INCIDENTS (>80%)
{'='*70}
"""
    
    # Add high-severity incidents
    high_sev = df[df['severity'] > 80].sort_values('severity', ascending=False)
    
    if len(high_sev) > 0:
        for _, row in high_sev.iterrows():
            report += f"""
Time: {row['timestamp']}
Student ID: {row['person_id']:03d}
Violation: {row['anomaly_type']}
Severity: {row['severity']:.1f}%
Description: {row['description']}
{'-'*70}
"""
    else:
        report += "\nNo high-severity incidents detected.\n"
    
    report += f"""
{'='*70}
PER-STUDENT ANALYSIS
{'='*70}
"""
    
    # Per-student stats
    student_stats = df.groupby('person_id').agg({
        'anomaly_type': 'count',
        'severity': ['mean', 'max']
    })
    
    for person_id, stats in student_stats.iterrows():
        count = stats[('anomaly_type', 'count')]
        avg_sev = stats[('severity', 'mean')]
        max_sev = stats[('severity', 'max')]
        
        report += f"Student {person_id:03d}: {count} violations, "
        report += f"Avg Severity: {avg_sev:.1f}%, Max: {max_sev:.1f}%\n"
    
    report += f"""
{'='*70}
RECOMMENDATIONS
{'='*70}
"""
    
    # Add recommendations based on findings
    if len(high_sev) > 0:
        report += "• Review high-severity incidents with video evidence\n"
    
    if df['person_id'].nunique() > 5:
        report += "• Multiple students flagged - consider general review\n"
    
    if 'head_turn' in df['anomaly_type'].values:
        report += "• Head turning detected - check seating arrangement\n"
    
    if 'possible_speaking' in df['anomaly_type'].values:
        report += "• Possible verbal communication detected\n"
    
    report += f"\n{'='*70}\n"
    report += "END OF REPORT\n"
    report += f"{'='*70}\n"
    
    # Save report
    with open('output/exam_report.txt', 'w') as f:
        f.write(report)
    
    print("Report saved to: output/exam_report.txt")
    print(report)


# ============================================================================
# MAIN - Run Examples
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("Exam Proctoring System - Usage Examples")
    print("="*60)
    print("Choose an example to run:")
    print("1. Basic Processing")
    print("2. Custom Configuration")
    print("3. Batch Processing")
    print("4. Analyze CSV Results")
    print("5. Real-time Monitoring")
    print("6. External Integration")
    print("7. Generate Report")
    print("="*60)
    
    choice = input("Enter choice (1-7): ").strip()
    
    examples = {
        '1': example_basic_processing,
        '2': example_custom_config,
        '3': example_batch_processing,
        '4': example_analyze_csv,
        '5': example_realtime_monitoring,
        '6': example_external_integration,
        '7': example_generate_report
    }
    
    if choice in examples:
        print(f"\nRunning example {choice}...\n")
        examples[choice]()
    else:
        print("Invalid choice. Run with: python examples.py")
        print("Then enter a number 1-7")