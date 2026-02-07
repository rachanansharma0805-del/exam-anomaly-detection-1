"""
Streamlit Dashboard for Exam Proctoring System - PRODUCTION READY
Features:
- Single invigilator detection display
- Consistent person ID tracking
- Excessive violation warnings
- Enhanced visualizations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import tempfile
from datetime import datetime
import cv2
import time
import traceback

from video_processor import ExamProctorPipeline


# Page configuration
st.set_page_config(
    page_title="Exam Proctoring AI - Production",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'output_video_path' not in st.session_state:
        st.session_state.output_video_path = None
    if 'csv_path' not in st.session_state:
        st.session_state.csv_path = None
    if 'summary_stats' not in st.session_state:
        st.session_state.summary_stats = None
    if 'df_anomalies' not in st.session_state:
        st.session_state.df_anomalies = None


def display_header():
    """Display dashboard header"""
    st.markdown('<div class="main-header">🎓 Exam Proctoring AI System</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Production-Ready Anomaly Detection with 95%+ Confidence</div>', 
                unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
        <h3>🎯 One Invigilator</h3>
        <p>Detects walking/standing person</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
        <h3>🔢 Consistent IDs</h3>
        <p>Same person = Same ID</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
        <h3>⚠️ Smart Alerts</h3>
        <p>Warns excessive violations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
        <h3>🔒 Privacy First</h3>
        <p>Automatic face blurring</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")


def upload_video_section():
    """Video upload section"""
    st.sidebar.header("📹 Video Upload")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload Exam Video",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file of the exam hall"
    )
    
    return uploaded_file


def processing_settings():
    """Processing settings section"""
    st.sidebar.header("⚙️ Detection Settings")
    
    settings = {
        'confidence_threshold': st.sidebar.slider(
            "Person Detection Confidence",
            min_value=0.3,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Minimum confidence for detecting people"
        ),
        'blur_faces': st.sidebar.checkbox(
            "🔒 Privacy Protection (Blur Faces)",
            value=True,
            help="Automatically blur all faces for GDPR compliance"
        ),
        'save_output_video': st.sidebar.checkbox(
            "Save Annotated Video",
            value=True,
            help="Save video with bounding boxes and alerts"
        ),
        'save_csv': st.sidebar.checkbox(
            "Save Detailed Log (CSV)",
            value=True,
            help="Export anomaly log for analysis"
        )
    }
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Detection Criteria:**
    
    ✓ **Invigilator:** Standing, walking, legs visible
    
    ✓ **Anomalies (95%+ confidence):**
    - Head turning (copying)
    - Hand raising (signaling)
    - Looking away
    - Speaking/communication
    - Body turning
    - Peeping at neighbors
    - Excessive movement
    
    ⚠️ **Excessive Violations:** 15+ anomalies = WARNING
    """)
    
    return settings


def process_video(video_file, settings):
    """Process uploaded video"""
    video_path = None
    pipeline = None
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            video_path = tmp_file.name
        
        # Initialize pipeline
        pipeline = ExamProctorPipeline(
            output_dir="output",
            confidence_threshold=settings['confidence_threshold'],
            blur_faces=settings['blur_faces']
        )
        
        # Create output paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video_path = f"output/exam_annotated_{timestamp}.mp4" if settings['save_output_video'] else None
        csv_path = f"output/anomalies_{timestamp}.csv" if settings['save_csv'] else None
        
        # Process with progress
        with st.spinner("🔄 Processing video... AI is analyzing the exam hall..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Initializing AI models...")
            progress_bar.progress(10)
            
            output_video, csv_file, summary = pipeline.process_video(
                video_path,
                output_video_path,
                csv_path
            )
            
            progress_bar.progress(100)
            status_text.success("✅ Processing complete!")
            
            # Cleanup
            pipeline.cleanup()
            
            try:
                cv2.destroyAllWindows()
            except:
                pass
            
            time.sleep(0.3)
            
            # Delete temp file
            try:
                if video_path and os.path.exists(video_path):
                    os.unlink(video_path)
            except:
                pass
            
            return output_video, csv_file, summary
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        
        with st.expander("Show detailed error"):
            st.code(traceback.format_exc())
        
        # Cleanup on error
        if pipeline:
            try:
                pipeline.cleanup()
            except:
                pass
        
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        if video_path and os.path.exists(video_path):
            try:
                time.sleep(0.5)
                os.unlink(video_path)
            except:
                pass
        
        return None, None, None


def display_summary_stats(stats):
    """Display summary statistics with warnings"""
    st.header("📊 Detection Summary")
    
    # Warning for excessive violations
    if stats.get('warned_students') and len(stats['warned_students']) > 0:
        st.markdown(f"""
        <div class="danger-box">
        <h3>⚠️ ALERT: Excessive Violations Detected</h3>
        <p><strong>{len(stats['warned_students'])} student(s)</strong> flagged with excessive anomalies (15+ violations)</p>
        <p><strong>Person IDs:</strong> {', '.join(map(str, stats['warned_students']))}</p>
        <p>These students require immediate review!</p>
        </div>
        """, unsafe_allow_html=True)
    elif stats['total_anomalies'] == 0:
        st.markdown("""
        <div class="success-box">
        <h3>✅ All Clear!</h3>
        <p>No anomalies detected. All students behaved appropriately.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Total Anomalies",
            value=stats['total_anomalies'],
            help="High-confidence violations (95%+)"
        )
    
    with col2:
        st.metric(
            label="Students Monitored",
            value=stats['total_students'],
            help="Unique students detected"
        )
    
    with col3:
        invigilator_status = "✓ Detected" if stats.get('invigilator_id') else "✗ Not Found"
        st.metric(
            label="Invigilator",
            value=invigilator_status,
            help=f"ID: {stats.get('invigilator_id', 'N/A')}"
        )
    
    with col4:
        st.metric(
            label="⚠️ Warned Students",
            value=len(stats.get('warned_students', [])),
            help="Students with 15+ violations"
        )
    
    with col5:
        avg_severity = stats.get('avg_severity', 0)
        severity_emoji = "🔴" if avg_severity > 70 else "🟡" if avg_severity > 40 else "🟢"
        st.metric(
            label=f"{severity_emoji} Avg Severity",
            value=f"{avg_severity:.1f}%",
            help="Average severity of violations"
        )
    
    st.markdown("---")


def display_anomaly_breakdown(stats):
    """Display anomaly type breakdown"""
    st.header("📋 Anomaly Analysis")
    
    if stats['total_anomalies'] == 0:
        return
    
    anomaly_types = stats.get('anomaly_types', {})
    
    if anomaly_types:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Pie chart
            fig_pie = px.pie(
                values=list(anomaly_types.values()),
                names=list(anomaly_types.keys()),
                title="Distribution of Violations",
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=list(anomaly_types.keys()),
                    y=list(anomaly_types.values()),
                    marker_color='indianred'
                )
            ])
            fig_bar.update_layout(
                title="Violation Counts by Type",
                xaxis_title="Violation Type",
                yaxis_title="Count",
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)


def display_detailed_log(df):
    """Display detailed anomaly log with filtering"""
    st.header("📝 Detailed Violation Log")
    
    if df is None or len(df) == 0:
        st.info("No violations detected")
        return
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        anomaly_filter = st.multiselect(
            "Filter by Violation Type",
            options=sorted(df['anomaly_type'].unique().tolist()),
            default=sorted(df['anomaly_type'].unique().tolist())
        )
    
    with col2:
        person_filter = st.multiselect(
            "Filter by Person ID",
            options=sorted(df['person_id'].unique().tolist()),
            default=sorted(df['person_id'].unique().tolist())
        )
    
    with col3:
        min_severity = st.slider(
            "Minimum Severity",
            min_value=0,
            max_value=100,
            value=0,
            step=10
        )
    
    with col4:
        show_excessive_only = st.checkbox(
            "⚠️ Excessive Violations Only",
            value=False,
            help="Show only students with 15+ violations"
        )
    
    # Apply filters
    filtered_df = df[
        (df['anomaly_type'].isin(anomaly_filter)) &
        (df['person_id'].isin(person_filter)) &
        (df['severity'] >= min_severity)
    ]
    
    if show_excessive_only:
        filtered_df = filtered_df[filtered_df['excessive_violations'] == True]
    
    st.write(f"**Showing {len(filtered_df)} of {len(df)} violations** (all with 95%+ confidence)")
    
    # Format DataFrame
    display_df = filtered_df[[
        'timestamp', 'person_id', 'anomaly_type', 
        'severity', 'description', 'confidence', 'excessive_violations'
    ]].copy()
    
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
    display_df['excessive_violations'] = display_df['excessive_violations'].apply(
        lambda x: '⚠️ YES' if x else 'No'
    )
    
    # Severity coloring
    def severity_color(val):
        if isinstance(val, str):
            return ''
        if val >= 70:
            return 'background-color: #ffcccc'
        elif val >= 40:
            return 'background-color: #ffe6cc'
        else:
            return 'background-color: #ccffcc'
    
    styled_df = display_df.style.applymap(severity_color, subset=['severity'])
    
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Timeline visualization
    if len(filtered_df) > 0:
        st.subheader("Timeline of Violations")
        
        fig_timeline = px.scatter(
            filtered_df,
            x='frame_number',
            y='person_id',
            color='anomaly_type',
            size='severity',
            hover_data=['timestamp', 'description', 'confidence'],
            title="Violation Timeline - When and Who",
            labels={'frame_number': 'Video Timeline (Frame)', 'person_id': 'Student ID'}
        )
        
        fig_timeline.update_layout(height=400)
        st.plotly_chart(fig_timeline, use_container_width=True)


def display_video_playback(video_path):
    """Display annotated video"""
    st.header("🎬 Annotated Video")
    
    if video_path and os.path.exists(video_path):
        try:
            file_size = os.path.getsize(video_path) / (1024 * 1024)
            
            if file_size > 200:
                st.warning(f"⚠️ Large file ({file_size:.1f} MB). Download recommended for better playback.")
            
            with open(video_path, 'rb') as video_file:
                video_bytes = video_file.read()
            
            st.video(video_bytes, format='video/mp4', start_time=0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download Annotated Video",
                    data=video_bytes,
                    file_name=os.path.basename(video_path),
                    mime="video/mp4",
                    use_container_width=True
                )
            
            with col2:
                st.info(f"📊 Size: {file_size:.1f} MB | 🔒 Faces blurred")
            
        except Exception as e:
            st.error(f"Error loading video: {str(e)}")


def display_csv_download(csv_path):
    """Display CSV download"""
    if csv_path and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            csv_data = df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Detailed CSV Log",
                data=csv_data,
                file_name=os.path.basename(csv_path),
                mime="text/csv",
                use_container_width=True
            )
            
            st.success(f"✓ CSV contains {len(df)} violation records")
        except Exception as e:
            st.error(f"Error: {str(e)}")


def main():
    """Main application"""
    initialize_session_state()
    display_header()
    
    # Sidebar
    uploaded_file = upload_video_section()
    settings = processing_settings()
    
    # Process button
    if uploaded_file is not None:
        st.sidebar.markdown("---")
        if st.sidebar.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
            output_video, csv_file, summary = process_video(uploaded_file, settings)
            
            if output_video or csv_file:
                st.session_state.processed = True
                st.session_state.output_video_path = output_video
                st.session_state.csv_path = csv_file
                st.session_state.summary_stats = summary
                
                if csv_file and os.path.exists(csv_file):
                    try:
                        st.session_state.df_anomalies = pd.read_csv(csv_file)
                    except:
                        st.session_state.df_anomalies = None
    
    # Display results
    if st.session_state.processed:
        if st.session_state.summary_stats:
            display_summary_stats(st.session_state.summary_stats)
        
        if st.session_state.summary_stats:
            display_anomaly_breakdown(st.session_state.summary_stats)
        
        st.markdown("---")
        
        if st.session_state.df_anomalies is not None:
            display_detailed_log(st.session_state.df_anomalies)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.output_video_path:
                display_video_playback(st.session_state.output_video_path)
        
        with col2:
            st.header("📊 Export Data")
            if st.session_state.csv_path:
                display_csv_download(st.session_state.csv_path)
    
    else:
        # Instructions
        st.markdown("""
        ## 📖 How It Works
        
        ### 1️⃣ Upload
        Upload your exam hall video using the sidebar
        
        ### 2️⃣ Configure
        - Set detection confidence (higher = fewer false positives)
        - Enable privacy protection (face blurring)
        - Choose output formats
        
        ### 3️⃣ Process
        Click "Start AI Analysis" - the system will:
        - ✓ Detect exactly ONE invigilator (walking/standing person)
        - ✓ Track students with consistent IDs
        - ✓ Monitor for violations with 95%+ confidence
        - ✓ Flag excessive violations (15+ anomalies)
        
        ### 4️⃣ Review
        - View detection summary
        - Analyze violation patterns
        - Review timeline and hotspots
        - Download annotated video and CSV log
        
        ---
        
        ## 🎯 What We Detect
        
        | Violation | Description | Severity |
        |-----------|-------------|----------|
        | 🔄 Head Turn | Turning to look at neighbor's paper | High |
        | ✋ Hand Raise | Raising hand (signaling or passing objects) | Medium |
        | 👀 Looking Away | Not focused on own paper | Medium |
        | 💬 Speaking | Communication with other students | High |
        | 🔃 Body Turn | Twisting body toward neighbor | High |
        | 👁️ Peeping | Leaning to view neighbor's work | Very High |
        | 🏃 Excessive Movement | Suspicious fidgeting/activity | Medium |
        
        **Excessive Violations:** 15+ anomalies triggers ⚠️ WARNING
        
        ---
        
        ## 🔒 Privacy & Compliance
        
        - ✓ Automatic face blurring (GDPR compliant)
        - ✓ No facial recognition
        - ✓ Anonymous student IDs
        - ✓ Secure local processing
        
        ---
        
        ## 💡 Pro Tips
        
        - Use high-quality video for best results
        - Ensure good lighting in exam hall
        - Position camera to capture full room
        - Review flagged students manually
        """)


if __name__ == "__main__":
    main()