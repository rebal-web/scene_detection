

import cv2
import torch
from scenedetect import VideoManager, SceneManager, frame_timecode
from scenedetect.detectors import ContentDetector
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

def detect_scenes(video_path):
    # Scene detection setup
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30))
    
    try:
        video_manager.set_downscale_factor(2)
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        return scene_manager.get_scene_list(), video_manager
        
    finally:
        video_manager.release()

def extract_frame(video_manager, timecode):
    """Extract frame at specified timecode"""
    video_manager.seek(timecode)
    frame = video_manager.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def initialize_caption_model():
    """Initialize BLIP model with CLIP Vision backbone"""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_summary(processor, model, image):
    """Generate caption using CLIP-based model"""
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True)

def analyze_video(video_path):
    # Initialize captioning model
    processor, model = initialize_caption_model()
    
    # Detect scenes
    scene_list, video_manager = detect_scenes(video_path)
    video_manager = VideoManager([video_path])
    video_manager.start()

    # Get FPS using OpenCV
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    results = []
    
    for i, scene in enumerate(scene_list):
        start_time, end_time = scene
        duration = end_time.get_seconds() - start_time.get_seconds()
        
        # Extract frame at scene midpoint
        midpoint_time = start_time.get_seconds() + (duration / 2)
        midpoint = int(midpoint_time * fps)
        frame = extract_frame(video_manager, midpoint)
        frame_image = Image.fromarray(frame)
        
        # Generate scene summary
        summary = generate_summary(processor, model, frame_image)
        
        results.append({
            "scene": i+1,
            "start": start_time.get_seconds(),
            "end": end_time.get_seconds(),
            "duration": duration,
            "summary": summary
        })
    
    video_manager.release()
    return results

if __name__ == "__main__":
    video_path = r"C:\Users\REEBAL\OneDrive\Desktop\nature.mp4"  # Replace with your video file
    
    # Analyze video and print results
    analysis_results = analyze_video(video_path)
    
    print("\nScene Analysis Results:")
    print("=======================")
    for result in analysis_results:
        print(f"Scene {result['scene']}:")
        print(f"  Time: {result['start']:.2f}s - {result['end']:.2f}s")
        print(f"  Duration: {result['duration']:.2f} seconds")
        print(f"  Summary: {result['summary']}")
        print()