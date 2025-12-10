import streamlit as st
import cv2
import tempfile
import os
from swing_analyzer import SwingAnalyzer
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Golf Swing Analyzer",
    page_icon="⛳",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f8b4c;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feedback-high {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
        margin: 0.5rem 0;
        color: #000000;
    }
    .feedback-medium {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 0.5rem 0;
        color: #000000;
    }
    .feedback-low {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 0.5rem 0;
        color: #000000;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">⛳ Golf Swing Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload your golf swing video and get AI-powered feedback</p>', unsafe_allow_html=True)
    
    # Sidebar with instructions
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. **Record your swing**: Use your phone to record a golf swing video
        2. **Upload the video**: Click the upload button and select your video
        3. **Get feedback**: Wait for the AI to analyze your swing
        4. **Improve**: Follow the suggestions to improve your technique
        
        **Tips for best results:**
        - Ensure full body is visible
        - Use good lighting
        - Film from the side view
        - Keep camera steady
        - Keep video under 30 seconds
        """)
        
        st.header("🎯 What we analyze")
        st.markdown("""
        - **Arm position and extension**
        - **Knee flex and stability**
        - **Spine angle**
        - **Shoulder rotation**
        - **Overall posture**
        """)
    
    # Main content area
    uploaded_file = st.file_uploader(
        "Upload your golf swing video",
        type=['mp4', 'mov', 'avi', 'mkv'],
        help="Supported formats: MP4, MOV, AVI, MKV"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Display original video
        st.subheader("📹 Your Original Video")
        st.video(uploaded_file)
        
        # Analyze button
        if st.button("🔍 Analyze My Swing", type="primary", use_container_width=True):
            with st.spinner("Analyzing your swing... This may take a minute ⏳"):
                try:
                    # Initialize analyzer
                    analyzer = SwingAnalyzer()
                    
                    # Process video
                    annotated_frames, all_metrics, all_feedback = analyzer.process_video(video_path)
                    
                    if not annotated_frames:
                        st.error("Could not process video. Please ensure the video is valid.")
                        return
                    
                    # Create output video with pose overlay
                    st.subheader("🎯 Analyzed Video with Pose Detection")
                    
                    # Save annotated video with browser-compatible codec
                    output_path = tempfile.mktemp(suffix='.mp4')
                    height, width = annotated_frames[0].shape[:2]
                    
                    # Try H264 codec first (most compatible), fallback to mp4v
                    fourcc_options = [
                        cv2.VideoWriter_fourcc(*'avc1'),  # H264 - best browser support
                        cv2.VideoWriter_fourcc(*'H264'),  # Alternative H264
                        cv2.VideoWriter_fourcc(*'X264'),  # Another H264 variant
                        cv2.VideoWriter_fourcc(*'mp4v'),  # Fallback
                    ]
                    
                    out = None
                    for fourcc in fourcc_options:
                        out = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height))
                        if out.isOpened():
                            break
                        out.release()
                    
                    if out and out.isOpened():
                        for frame in annotated_frames:
                            out.write(frame)
                        out.release()
                        
                        # Display annotated video
                        try:
                            with open(output_path, 'rb') as video_file:
                                video_bytes = video_file.read()
                                st.video(video_bytes)
                        except Exception as e:
                            st.error(f"Could not display video: {str(e)}")
                            st.info("Video was processed but display failed. See individual frames below.")
                    else:
                        st.warning("Could not create video file. Showing sample frames instead.")
                        
                        # Show sample frames as fallback
                        st.subheader("Sample Analyzed Frames")
                        cols = st.columns(3)
                        sample_indices = [0, len(annotated_frames)//2, -1]
                        for idx, col in enumerate(cols):
                            if idx < len(sample_indices):
                                frame_idx = sample_indices[idx]
                                with col:
                                    st.image(cv2.cvtColor(annotated_frames[frame_idx], cv2.COLOR_BGR2RGB),
                                            caption=f"Frame {frame_idx + 1}", use_container_width=True)
                    
                    # Generate summary feedback
                    st.subheader("📊 Analysis Results")
                    
                    if all_metrics:
                        summary_feedback = analyzer.get_summary_feedback(all_metrics, all_feedback)
                        
                        # Display feedback cards
                        for fb in summary_feedback:
                            severity_class = f"feedback-{fb['severity']}"
                            severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[fb['severity']]
                            
                            st.markdown(f"""
                            <div class="{severity_class}">
                                <h3>{severity_emoji} {fb['issue']}</h3>
                                <p><strong>Analysis:</strong> {fb['description']}</p>
                                <p><strong>💡 Tip:</strong> {fb['tip']}</p>
                                {f"<p><small>{fb.get('frequency', '')}</small></p>" if 'frequency' in fb else ""}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Display metrics
                        st.subheader("📈 Key Metrics (Average)")
                        
                        if all_metrics:
                            avg_metrics = {
                                key: np.mean([m[key] for m in all_metrics])
                                for key in all_metrics[0].keys()
                            }
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Lead Arm Angle", f"{avg_metrics['left_arm_angle']:.1f}°")
                                st.metric("Trail Arm Angle", f"{avg_metrics['right_arm_angle']:.1f}°")
                            
                            with col2:
                                st.metric("Lead Knee Flex", f"{avg_metrics['left_knee_angle']:.1f}°")
                                st.metric("Trail Knee Flex", f"{avg_metrics['right_knee_angle']:.1f}°")
                            
                            with col3:
                                st.metric("Spine Angle", f"{abs(avg_metrics['spine_angle']):.1f}°")
                                st.metric("Shoulder Rotation", f"{abs(avg_metrics['shoulder_rotation']):.1f}°")
                    else:
                        st.warning("Could not extract metrics from the video. Please ensure your full body is visible.")
                    
                    # Success message
                    st.success("✅ Analysis complete! Review the feedback above to improve your swing.")
                    
                    # Cleanup
                    os.unlink(output_path)
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
                    st.info("Please try with a different video or ensure the video quality is good.")
        
        # Cleanup temp file
        os.unlink(video_path)
    
    else:
        # Show example/placeholder when no video is uploaded
        st.info("👆 Upload a video of your golf swing to get started!")
        
        # Show sample feedback
        with st.expander("📖 See example feedback"):
            st.markdown("""
            <div class="feedback-high">
                <h3>🔴 Bent Lead Arm</h3>
                <p><strong>Analysis:</strong> Your lead arm is bending too much (145.3°). Try to keep it straighter throughout the swing for better consistency and power transfer.</p>
                <p><strong>💡 Tip:</strong> Focus on maintaining arm extension from setup through impact.</p>
            </div>
            
            <div class="feedback-medium">
                <h3>🟡 Spine Angle</h3>
                <p><strong>Analysis:</strong> Your spine tilt (48.2°) might be too extreme.</p>
                <p><strong>💡 Tip:</strong> Reduce forward bend slightly for better balance and rotation.</p>
            </div>
            
            <div class="feedback-low">
                <h3>🟢 Trail Arm Position</h3>
                <p><strong>Analysis:</strong> Your trail arm angle (162.5°) is in good position.</p>
                <p><strong>💡 Tip:</strong> Keep this consistent!</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()