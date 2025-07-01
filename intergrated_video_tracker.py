import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18
from torch import nn

# Import your data preprocessing class
from data_preprocessing import SmileDatasetProcessor

class SmileVideoProcessor:
    """
    Enhanced video processor using your data preprocessing pipeline
    """
    
    def __init__(self, model_path="smile_resnet18.pt", data_processor=None):
        """
        Initialize video processor with your data preprocessing integration
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_processor = data_processor or SmileDatasetProcessor()
        
        # Load trained model
        self.model = self._load_model(model_path)
        
        # Setup transforms (compatible with your preprocessing)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # Match training size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Setup face detection
        self.net = cv2.dnn.readNetFromCaffe(
            "deploy.prototxt",
            "res10_300x300_ssd_iter_140000.caffemodel"
        )
    
    def _load_model(self, model_path):
        """Load the trained smile detection model"""
        model = resnet18()
        model.fc = nn.Linear(model.fc.in_features, 2)
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Loaded model from {model_path}")
        else:
            print(f"⚠️  Model {model_path} not found. Using random weights.")
        
        model.to(self.device).eval()
        return model
    
    def preprocess_face_for_inference(self, face_crop):
        """
        Use your preprocessing pipeline for consistent inference
        """
        # Convert BGR to RGB (consistent with your preprocessing)
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        # Apply transform
        input_tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)
        return input_tensor
    
    def detect_smile(self, face_crop, threshold=0.5):
        """
        Detect smile using your preprocessing pipeline
        """
        # Preprocess using your method
        input_tensor = self.preprocess_face_for_inference(face_crop)
        
        # Inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)
            smile_prob = probs[0, 1].item()
        
        return smile_prob > threshold, smile_prob
    
    def process_video(self, video_path, output_path="output_with_preprocessing.avi"):
        """
        Process video with smile detection using your preprocessing pipeline
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return
        
        # Video properties
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
        FRAME_TIME = 1.0 / FPS
        
        # Output writer
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"XVID"),
            FPS,
            (W, H)
        )
        
        # Tracking variables
        face_time = 0.0
        smile_time = 0.0
        frame_count = 0
        
        print(f"Processing video: {video_path}")
        print("Press 'q' to quit early")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Face detection
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0)
            )
            self.net.setInput(blob)
            detections = self.net.forward()
            
            # Find best face
            best_face = None
            best_confidence = 0
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5 and confidence > best_confidence:
                    x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * [W, H, W, H]).astype(int)
                    best_face = (x1, y1, x2, y2)
                    best_confidence = confidence
            
            if best_face:
                x1, y1, x2, y2 = best_face
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    # Use your preprocessing for smile detection
                    is_smiling, smile_prob = self.detect_smile(face_crop)
                    
                    # Update timers
                    face_time += FRAME_TIME
                    if is_smiling:
                        smile_time += FRAME_TIME
                    
                    # Draw annotations
                    color = (0, 255, 0) if is_smiling else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add text
                    text = f"{'Smiling' if is_smiling else 'Not Smiling'} ({smile_prob:.2f})"
                    cv2.putText(frame, text, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Add timing info
                    engagement = (smile_time / face_time * 100) if face_time > 0 else 0
                    timing_text = f"Face: {face_time:.1f}s | Smile: {smile_time:.1f}s | Engagement: {engagement:.1f}%"
                    cv2.putText(frame, timing_text, (10, H-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Write frame
            out.write(frame)
            
            # Display (optional)
            cv2.imshow("Smile Detection with Preprocessing", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Progress update
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Final report using your processor's reporting
        print(f"\n{'='*50}")
        print("VIDEO PROCESSING REPORT")
        print(f"{'='*50}")
        print(f"Total frames processed: {frame_count}")
        print(f"Face detection time: {face_time:.3f} seconds")
        print(f"Smile duration: {smile_time:.3f} seconds")
        if face_time > 0:
            print(f"Engagement score: {smile_time/face_time*100:.1f}%")
        print(f"Output saved: {output_path}")
        print(f"{'='*50}")

def main():
    """
    Main function demonstrating integration
    """
    print("=== INTEGRATED SMILE VIDEO PROCESSING ===")
    
    # Initialize your data processor
    processor = SmileDatasetProcessor(data_dir='./smile_dataset')
    
    # Initialize video processor with your preprocessing
    video_processor = SmileVideoProcessor(
        model_path="smile_resnet18.pt",
        data_processor=processor
    )
    
    # Process video
    video_path = input("Enter video path (or press Enter for 'Smiling_Video.mp4'): ").strip()
    if not video_path:
        video_path = "Smiling_Video.mp4"
    
    if os.path.exists(video_path):
        video_processor.process_video(video_path)
    else:
        print(f"Video file not found: {video_path}")
        print("Please ensure the video file exists in the current directory.")

if __name__ == "__main__":
    main()