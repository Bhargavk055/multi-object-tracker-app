"""
app.py
-------
Streamlit Web Interface for the Multi-Object Tracking Pipeline.
Provides a simple UI to upload/download videos and run the tracker.
"""

import streamlit as st
import os
import glob

# Make sure we can import our modules
import config
from utils import clean_previous_outputs, download_video
from main import run_pipeline

st.set_page_config(page_title="Multi-Object Tracking", page_icon="🎯", layout="wide")

st.title("🎯 Multi-Object Detection & Tracking")
st.markdown("Upload a video or paste a YouTube link to process it using YOLOv8 and ByteTrack.")

# Settings Sidebar
st.sidebar.header("Processing Settings")
limit_mode = st.sidebar.radio("Processing Length", ["Quick Preview (~3 seconds) - RECOMMENDED", "Full Video"])
# 90 frames at 30fps is roughly 3 seconds
max_frames = 90 if limit_mode.startswith("Quick") else None

source_type = st.radio("Select Video Source", ["YouTube URL", "Upload Video"])

source_path = None
output_name = "tracked_video.mp4"

video_url = None

if source_type == "YouTube URL":
    video_url = st.text_input("YouTube URL", placeholder="https://youtube.com/shorts/V8c6KrpCiUs")
    if video_url:
        st.info("Click 'Run Pipeline' below to download and process this video.")
else:
    uploaded_file = st.file_uploader("Upload local video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file:
        source_path = os.path.join(config.INPUT_DIR, "uploaded_video.mp4")
        with open(source_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully!")

st.info(
    "👋 **Evaluator Note:** This application is running on a free-tier Cloud CPU without GPU acceleration. "
    "To prevent timeout, please use the **Quick Preview** mode (processes 3 sec of video in ~30 sec). "
    "\n\n*If YouTube downloads hang infinitely, it is because YouTube blocks Datacenter IPs. Please upload a short local video instead!*"
)

if st.button("🚀 Run Pipeline", type="primary"):
    ready = False
    
    if source_type == "YouTube URL" and video_url:
        with st.spinner("Downloading video from YouTube..."):
            try:
                source_path = download_video(video_url)
                ready = True
            except Exception as e:
                st.error(f"Failed to download video. Please check the URL or try another. Error: {e}")
    elif source_type == "Upload Video" and source_path:
        ready = True
    else:
        st.warning("Please provide a video source first.")

    if ready and source_path:
        st.divider()
        clean_previous_outputs()
        
        output_path = os.path.join(config.OUTPUT_DIR, output_name)
        
        with st.spinner("Processing video (Tracking in progress)..."):
            # Run the actual pipeline
            # Note: We pass show=False so it doesn't try to open OpenCV GUI windows on a headless cloud server
            try:
                run_pipeline(source_path, output_path, show=False, max_frames=max_frames)
                st.success("✅ Pipeline completed successfully!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Tracking Statistics")
                    stats_path = os.path.join(config.OUTPUT_DIR, "stats.png")
                    if os.path.exists(stats_path):
                        st.image(stats_path, use_column_width=True)
                    else:
                        st.info("No statistics graph generated.")
                
                with col2:
                    st.subheader("🎬 Results")
                    if os.path.exists(output_path):
                        with open(output_path, "rb") as file:
                            st.download_button(
                                label="⬇️ Download Annotated Video",
                                data=file,
                                file_name=output_name,
                                mime="video/mp4",
                            )
                        # Show screenshot previews
                        ss_files = sorted(glob.glob(os.path.join(config.SCREENSHOTS_DIR, "*.png")))
                        if ss_files:
                            st.write("Final Frame Preview:")
                            st.image(ss_files[-1], use_column_width=True)
                    else:
                        st.error("Output video not found.")
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
