# app.py - Military Drone Detection Streamlit App

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Military Drone Detection", page_icon="🚁", layout="wide")

@st.cache_resource
def load_model():
    model_path = "results/military_drone_model/weights/best.pt"
    if not os.path.exists(model_path):
        st.error("❌ Model not found! Train first: python main.py")
        return None
    st.success("✅ Model loaded")
    return YOLO(model_path)

def add_overlay_stats(img, boxes, inference_time):
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (310, 130), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "DETECTION STATUS", (20, 35), font, 0.7, (255, 255, 255), 2)
    
    y = 65
    color = (0, 255, 0) if len(boxes) > 0 else (128, 128, 128)
    
    if len(boxes) > 0:
        cv2.putText(img, f"Drones: {len(boxes)}", (20, y), font, 0.7, color, 2)
        avg_conf = float(boxes.conf.mean())
        cv2.putText(img, f"Confidence: {avg_conf:.1%}", (20, y + 30), font, 0.7, color, 2)
        cv2.putText(img, f"Time: {inference_time:.1f}ms", (20, y + 60), font, 0.7, color, 2)
    else:
        cv2.putText(img, "Drones: 0", (20, y), font, 0.7, color, 2)
        cv2.putText(img, "Status: CLEAR", (20, y + 30), font, 0.7, color, 2)
        cv2.putText(img, f"Time: {inference_time:.1f}ms", (20, y + 60), font, 0.7, color, 2)
    
    return img

def main():
    st.title("🚁 Military Drone Detection with YOLO11")
    st.markdown("**Target UAVs:** Shahed-136, Lancet, Orlan-10, ZALA, Forpost, and others")
    st.markdown("---")
    
    st.sidebar.title("Settings")
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.40, 0.05)
    st.sidebar.info("💡 Recommended: 0.35-0.45 to reduce false positives")
    
    model = load_model()
    if model is None:
        st.stop()
    
    tab1, tab2, tab3 = st.tabs(["📷 Image Upload", "🎥 Video Upload", "📹 Webcam"])
    
    with tab1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            
            with st.spinner("🔍 Detecting drones..."):
                results = model(image, conf=confidence)
                annotated_img = results[0].plot()
                annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                
                boxes = results[0].boxes
                inference_time = results[0].speed['inference']
                annotated_img = add_overlay_stats(annotated_img, boxes, inference_time)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            with col2:
                st.subheader("Detection Results")
                st.image(annotated_img, use_container_width=True)
            
            if len(boxes) > 0:
                st.markdown("---")
                st.subheader("🎯 Detection Details")
                for i, box in enumerate(boxes):
                    conf = float(box.conf[0])
                    st.write(f"**Drone {i+1}:** Confidence = {conf:.2%}")
            else:
                st.info("✅ No drones detected")
    
    with tab2:
        st.header("Upload Video")
        video_file = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov'], key="video_uploader")
        
        if video_file and st.button("🎬 Process Video"):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            
            with st.spinner("Processing video..."):
                cap = cv2.VideoCapture(tfile.name)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                output_path = "output_detection.mp4"
                # Try different codecs for compatibility
                fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                # If avc1 fails, fallback to XVID
                if not out.isOpened():
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    output_path = "output_detection.avi"
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                progress_bar = st.progress(0)
                frame_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    results = model(frame, conf=confidence, verbose=False)
                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)
                    
                    frame_count += 1
                    progress_bar.progress(frame_count / total_frames)
                
                cap.release()
                out.release()
                
                st.success("✅ Video processed!")
                st.video(output_path)
                
                with open(output_path, 'rb') as f:
                    st.download_button("📥 Download Video", f, "drone_detection.mp4", "video/mp4")
    
    with tab3:
        st.header("Real-time Webcam Detection")
        st.info("⚠️ Opens OpenCV window. Press 'q' to quit.")
        
        if st.button("📹 Start Webcam"):
            st.warning("Webcam opening...")
            cap = cv2.VideoCapture(0)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = model(frame, conf=confidence, verbose=False)
                annotated = results[0].plot()
                
                fps = 1000 / results[0].speed['inference']
                cv2.putText(annotated, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated, f'Drones: {len(results[0].boxes)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Drone Detection', annotated)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            st.success("✅ Webcam closed")

if __name__ == "__main__":
    main()