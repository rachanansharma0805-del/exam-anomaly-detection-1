# streamlit_app.py
import streamlit as st
import pandas as pd
import tempfile
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

from exam_detector import ExamAnomalyDetector

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Exam Hall Anomaly Detection",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = None
if "video_path" not in st.session_state:
    st.session_state.video_path = None

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/student-male.png", width=80)
    st.header("⚙️ Detection Settings")
    
    st.subheader("Thresholds")
    head_turn = st.slider(
        "Head Turn Sensitivity", 
        min_value=5, 
        max_value=30, 
        value=10,
        help="Lower values = more sensitive detection"
    )
    
    hand_move = st.slider(
        "Hand Movement Sensitivity", 
        min_value=10, 
        max_value=50, 
        value=25,
        help="Lower values = more sensitive detection"
    )
    
    sensitivity = st.select_slider(
        "Overall Sensitivity",
        options=["low", "medium", "high"],
        value="medium",
        help="Adjusts all detection thresholds"
    )
    
    st.subheader("Processing Options")
    skip_frames = st.number_input(
        "Skip Frames (for faster processing)",
        min_value=0,
        max_value=10,
        value=0,
        help="Skip N frames between analysis (0 = analyze all)"
    )
    
    clip_duration = st.slider(
        "Anomaly Clip Duration (seconds)",
        min_value=2,
        max_value=10,
        value=3
    )
    
    config = {
        "head_turn_threshold": head_turn,
        "hand_move_threshold": hand_move,
        "sensitivity": sensitivity,
        "skip_frames": skip_frames,
        "clip_duration": clip_duration
    }
    
    st.divider()
    st.caption("💡 Tip: Adjust sensitivity based on exam environment")

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("🎓 Exam Hall Anomaly Detection System")
st.markdown("**Real-time monitoring and analysis of suspicious activities during examinations**")

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📤 Upload & Analyze",
    "📊 Detection Results",
    "🎬 Anomaly Highlights",
    "📈 Analytics"
])

# =================================================
# TAB 1 — UPLOAD & ANALYZE
# =================================================
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Exam Video")
        uploaded_file = st.file_uploader(
            "Select a video file",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload exam hall surveillance footage"
        )
        
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_file.read())
                st.session_state.video_path = tmp.name
            
            st.success(f"✅ Video uploaded: {uploaded_file.name}")
            
            # Show video preview
            with st.expander("📹 Preview Video"):
                st.video(st.session_state.video_path)
    
    with col2:
        st.subheader("Detection Info")
        st.info("""
        **What we detect:**
        - 👤 Head turns & looking around
        - ✋ Suspicious hand movements
        - 📄 Potential paper exchanges
        - 📱 Electronic device usage
        - 🔄 Unusual behavior patterns
        """)
    
    st.divider()
    
    if uploaded_file:
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        
        with col_btn1:
            analyze_btn = st.button(
                "🚀 Start Detection", 
                type="primary",
                width='stretch'  # Updated from use_container_width=True
            )
        
        with col_btn2:
            if st.session_state.results:
                st.button(
                    "🔄 Reset",
                    on_click=lambda: st.session_state.update({"results": None}),
                    width='stretch'  # Updated from use_container_width=True
                )
        
        if analyze_btn:
            output_dir = tempfile.mkdtemp(prefix=r"C:\Users\rachana sharma\exam-hall-anomaly\Week 4")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            stats_cols = st.columns(3)
            
            metric_frame = stats_cols[0].empty()
            metric_fps = stats_cols[1].empty()
            metric_anomaly = stats_cols[2].empty()
            
            def progress_callback(curr, total, count):
                progress = curr / total
                progress_bar.progress(progress)
                status_text.text(f"🔍 Analyzing frame {curr:,} of {total:,}")
                
                metric_frame.metric("Frame", f"{curr:,}/{total:,}")
                if curr > 0:
                    metric_fps.metric("Processing FPS", f"{curr / ((curr/30)):.1f}")
                metric_anomaly.metric("Anomalies Detected", count)
            
            detector = ExamAnomalyDetector(config)
            
            with st.spinner("🔄 Processing video... This may take a few minutes."):
                try:
                    results = detector.process_video(
                        st.session_state.video_path,
                        output_dir,
                        progress_callback
                    )
                    
                    st.session_state.results = results
                    progress_bar.progress(1.0)
                    status_text.empty()
                    
                    st.success("✅ Analysis completed successfully!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")

# =================================================
# TAB 2 — DETECTION RESULTS
# =================================================
with tab2:
    if st.session_state.results:
        r = st.session_state.results
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Frames",
                f"{r['total_frames']:,}",
                help="Total number of frames in video"
            )
        
        with col2:
            st.metric(
                "Duration",
                f"{r['duration']:.1f}s",
                help="Video duration in seconds"
            )
        
        with col3:
            st.metric(
                "Anomalies",
                r['total_anomalies'],
                help="Total suspicious activities detected"
            )
        
        with col4:
            st.metric(
                "Processing Speed",
                f"{r['avg_processing_fps']:.1f} FPS",
                help="Average frames processed per second"
            )
        
        st.divider()
        
        # Anomaly breakdown
        if r['total_anomalies'] > 0:
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                st.subheader("📋 Detailed Anomaly Log")
                df = pd.DataFrame(r["anomaly_log"])
                
                # Add formatted columns
                df['Time'] = df['timestamp'].apply(lambda x: f"{int(x//60):02d}:{int(x%60):02d}")
                df['Confidence'] = df['confidence'].apply(lambda x: f"{x:.1f}%")
                
                # Reorder columns
                display_df = df[['Time', 'type', 'frame', 'Confidence', 'description']]
                display_df.columns = ['Time', 'Type', 'Frame', 'Confidence', 'Description']
                
                # Color code by type
                st.dataframe(
                    display_df,
                    width='stretch',  # Updated from use_container_width=True
                    height=400
                )
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download Full Report (CSV)",
                    csv,
                    "anomaly_report.csv",
                    "text/csv",
                    key='download-csv'
                )
            
            with col_right:
                st.subheader("📊 Anomaly Distribution")
                
                if 'anomaly_types' in r:
                    types_df = pd.DataFrame([
                        {"Type": k, "Count": v} 
                        for k, v in r['anomaly_types'].items()
                    ])
                    
                    fig = px.pie(
                        types_df, 
                        values='Count', 
                        names='Type',
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, width='stretch')  # Updated for consistency
                    
                    st.dataframe(
                        types_df.sort_values('Count', ascending=False),
                        width='stretch',  # Updated from use_container_width=True
                        hide_index=True
                    )
        else:
            st.success("🎉 No anomalies detected! Clean examination conduct.")
    
    else:
        st.info("👆 Upload and analyze a video in the 'Upload & Analyze' tab first")

# =================================================
# TAB 3 — ANOMALY HIGHLIGHTS
# =================================================
with tab3:
    if st.session_state.results:
        r = st.session_state.results
        
        if r["merged_video"] and os.path.exists(r["merged_video"]):
            st.subheader("🎬 Merged Anomaly Highlights")
            st.caption("This video contains all detected suspicious activities with timestamps and labels")
            
            # Check file size (critical for HTML embedding)
            file_size_mb = os.path.getsize(r["merged_video"]) / (1024 * 1024)
            if file_size_mb > 5:  # Browser data URI limit is ~2-5MB
                st.warning(f"⚠️ Video is too large ({file_size_mb:.1f} MB) for inline playback. Download and play locally instead.")
                # Skip embedding, go straight to download
                with open(r["merged_video"], "rb") as f:
                    video_bytes = f.read()
                    st.download_button(
                        "📥 Download Highlights Video",
                        video_bytes,
                        "anomaly_highlights.mp4",
                        "video/mp4"
                    )
            else:
                try:
                    import base64
                    
                    # Read video as bytes and encode to base64
                    with open(r["merged_video"], "rb") as f:
                        video_bytes = f.read()
                        video_b64 = base64.b64encode(video_bytes).decode()
                    
                    # Embed using HTML <video> tag with data URI
                    video_html = f"""
                    <video width="100%" controls>
                        <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                    """
                    st.markdown(video_html, unsafe_allow_html=True)
                    
                    # Download button (still available)
                    st.download_button(
                        "📥 Download Highlights Video",
                        video_bytes,
                        "anomaly_highlights.mp4",
                        "video/mp4"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error embedding video: {str(e)}")
                    st.warning("""
                    **Embedding Failed**: Try downloading the video and playing it locally.
                    
                    **Solutions:**
                    1. Install FFmpeg for better codec support.
                    2. Convert manually: `ffmpeg -i input.mp4 -c:v libx264 output.mp4`
                    """)
                    
                    # Fallback download
                    with open(r["merged_video"], "rb") as f:
                        st.download_button(
                            "📥 Download Video (Play Locally)",
                            f.read(),
                            "anomaly_highlights.mp4",
                            "video/mp4"
                        )
        else:
            st.success("🎉 No anomalies detected, so no highlight reel was generated.")
    else:
        st.info("👆 Run detection first to generate anomaly highlights")

# =================================================
# TAB 4 — ANALYTICS
# =================================================
with tab4:
    if st.session_state.results:
        r = st.session_state.results
        
        if r['total_anomalies'] > 0:
            df = pd.DataFrame(r["anomaly_log"])
            
            st.subheader("📈 Temporal Analysis")
            
            # Timeline chart
            fig_timeline = px.scatter(
                df,
                x='timestamp',
                y='type',
                color='type',
                size='confidence',
                hover_data=['frame', 'confidence', 'description'],
                title='Anomaly Timeline',
                labels={'timestamp': 'Time (seconds)', 'type': 'Anomaly Type'}
            )
            fig_timeline.update_layout(height=400)
            st.plotly_chart(fig_timeline, width='stretch')  # Updated for consistency
            
            # Confidence distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Confidence Distribution")
                fig_conf = px.histogram(
                    df,
                    x='confidence',
                    nbins=20,
                    title='Detection Confidence Levels',
                    labels={'confidence': 'Confidence (%)', 'count': 'Frequency'}
                )
                st.plotly_chart(fig_conf, width='stretch')  # Updated for consistency
            
            with col2:
                st.subheader("📊 Anomalies Over Time")
                
                # Create time bins
                df['time_bin'] = (df['timestamp'] // 30).astype(int) * 30
                time_counts = df.groupby('time_bin').size().reset_index(name='count')
                
                fig_over_time = px.bar(
                    time_counts,
                    x='time_bin',
                    y='count',
                    title='Anomaly Frequency (30s intervals)',
                    labels={'time_bin': 'Time (seconds)', 'count': 'Number of Anomalies'}
                )
                st.plotly_chart(fig_over_time, width='stretch')  # Updated for consistency
            
            # Summary statistics
            st.subheader("📋 Summary Statistics")
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                avg_conf = df['confidence'].mean()
                st.metric("Average Confidence", f"{avg_conf:.1f}%")
            
            with summary_col2:
                high_conf = len(df[df['confidence'] > 80])
                st.metric("High Confidence Detections", high_conf)
            
            with summary_col3:
                anomaly_rate = (r['total_anomalies'] / r['total_frames']) * 100
                st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
        
        else:
            st.info("No anomalies to analyze")
    else:
        st.info("👆 Run detection first to view analytics")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption("🤖 Powered by OpenCV & Computer Vision")

with footer_col2:
    st.caption("⚡ Real-time Motion Detection")

with footer_col3:
    st.caption("🔒 Secure Processing")