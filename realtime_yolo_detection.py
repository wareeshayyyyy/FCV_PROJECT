"""
Real-time Bone Fracture Detection using YOLO
Supports webcam, video file, and image file inputs
"""

import cv2
import argparse
import time
import os
from pathlib import Path
from ultralytics import YOLO
import numpy as np


class RealTimeBoneFractureDetector:
    """Real-time bone fracture detection using YOLO"""
    
    # Class names from the dataset
    CLASS_NAMES = [
        'elbow positive',
        'fingers positive', 
        'forearm fracture',
        'humerus fracture',
        'humerus',
        'shoulder fracture',
        'wrist positive'
    ]
    
    # Color mapping for different fracture types
    COLORS = [
        (0, 255, 0),      # Green - elbow
        (255, 0, 0),      # Blue - fingers
        (0, 0, 255),      # Red - forearm
        (255, 255, 0),    # Cyan - humerus fracture
        (255, 0, 255),    # Magenta - humerus
        (0, 255, 255),    # Yellow - shoulder
        (128, 0, 128)     # Purple - wrist
    ]
    
    def __init__(self, model_path=None, conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize the detector
        
        Args:
            model_path: Path to custom trained YOLO model (.pt file). 
                       If None, uses pretrained YOLOv8n
            conf_threshold: Confidence threshold for detections (0-1)
            iou_threshold: IoU threshold for NMS (0-1)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.is_custom_model = False
        
        print("Loading YOLO model...")
        
        # Try to find trained model if not specified
        if model_path is None:
            # Check for trained models in common locations
            possible_paths = [
                'yolo_training_results/yolov8n_bone_fracture/weights/best.pt',
                'yolo_training_results/yolov8s_bone_fracture/weights/best.pt',
                'yolo_training_results/yolov8m_bone_fracture/weights/best.pt',
                'checkpoints/yolo_best.pt',
                'runs/detect/train/weights/best.pt'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"📁 Found trained model: {path}")
                    break
        
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            self.is_custom_model = True
            print(f"✓ Loaded CUSTOM bone fracture model from: {model_path}")
            print("  This model detects: elbow, fingers, forearm, humerus, shoulder, wrist fractures")
            
            # Get model info
            if hasattr(self.model, 'info'):
                try:
                    info = self.model.info(verbose=False)
                    print(f"  Model parameters: {info.get('parameters', 'N/A')}")
                except:
                    pass
        else:
            # Use pretrained YOLOv8n (nano) for faster inference
            # You can change to 'yolov8s.pt', 'yolov8m.pt', etc. for better accuracy
            self.model = YOLO('yolov8n.pt')
            self.is_custom_model = False
            print("⚠ Using PRETRAINED YOLOv8n model (COCO dataset)")
            print("  This model detects GENERAL OBJECTS (people, cars, dogs, etc.), NOT bone fractures!")
            print("  To detect bone fractures, train a custom model:")
            print("     python train_yolo_proper.py")
            if model_path:
                print(f"  Warning: Custom model path '{model_path}' not found, using pretrained model")
        
        # Get class names from the model (will be COCO classes for pretrained, or custom for trained model)
        self.model_class_names = self.model.names if hasattr(self.model, 'names') else {}
        
        # Update class names if custom model (use bone fracture names)
        if self.is_custom_model:
            # Override with bone fracture class names
            self.CLASS_NAMES = [
                'elbow positive',
                'fingers positive', 
                'forearm fracture',
                'humerus fracture',
                'humerus',
                'shoulder fracture',
                'wrist positive'
            ]
        else:
            # Use model's class names (COCO)
            pass
        
        # Set model parameters
        self.model.overrides['conf'] = conf_threshold
        self.model.overrides['iou'] = iou_threshold
        
    def draw_detections(self, frame, results):
        """
        Draw bounding boxes and labels on the frame
        
        Args:
            frame: Input frame (numpy array)
            results: YOLO detection results
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Process each detection
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get class and confidence
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Get class name from model (COCO classes for pretrained, or custom classes)
                    if cls in self.model_class_names:
                        class_name = self.model_class_names[cls]
                    elif self.is_custom_model and cls < len(self.CLASS_NAMES):
                        # Fallback to bone fracture class names for custom models
                        class_name = self.CLASS_NAMES[cls]
                    else:
                        class_name = f"Class {cls}"
                    
                    # Get color for this class
                    # Use different colors for custom vs pretrained models
                    if self.is_custom_model:
                        color = self.COLORS[cls % len(self.COLORS)]
                    else:
                        # Use a standard color for COCO detections
                        color = (0, 255, 0)  # Green for general objects
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Prepare label text
                    label = f"{class_name}: {conf:.2f}"
                    
                    # Calculate text size for background
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # Draw label background
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1 - text_height - baseline - 5),
                        (x1 + text_width, y1),
                        color,
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y1 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
        
        return annotated_frame
    
    def detect_webcam(self, camera_id=0, save_output=False, output_path='output_video.mp4'):
        """
        Real-time detection from webcam
        
        Args:
            camera_id: Camera device ID (usually 0 for default webcam)
            save_output: Whether to save the output video
            output_path: Path to save output video
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer for saving output
        out = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("\nStarting real-time detection...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        frame_count = 0
        fps_start_time = time.time()
        fps_counter = 0
        current_fps = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to read frame from camera")
                    break
                
                # Perform detection
                results = self.model(frame, verbose=False)
                
                # Draw detections
                annotated_frame = self.draw_detections(frame, results)
                
                # Calculate FPS
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Display FPS and detection count
                detection_count = sum(len(r.boxes) if r.boxes is not None else 0 for r in results)
                model_type = "Bone Fracture Detection" if self.is_custom_model else "COCO Objects (NOT fractures!)"
                info_text = f"FPS: {current_fps} | Detections: {detection_count} | {model_type}"
                cv2.putText(
                    annotated_frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0) if self.is_custom_model else (0, 165, 255),  # Orange warning for pretrained
                    2
                )
                
                # Display frame
                cv2.imshow('Bone Fracture Detection - Real-time', annotated_frame)
                
                # Save frame if requested
                if save_output and out is not None:
                    out.write(annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_path = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_path, annotated_frame)
                    print(f"Screenshot saved: {screenshot_path}")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
            print("Detection stopped")
    
    def detect_video(self, video_path, save_output=False, output_path=None):
        """
        Detection on video file
        
        Args:
            video_path: Path to input video file
            save_output: Whether to save annotated video
            output_path: Path to save output video (auto-generated if None)
        """
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Generate output path if not provided
        if save_output and output_path is None:
            input_path = Path(video_path)
            output_path = str(input_path.parent / f"{input_path.stem}_detected{input_path.suffix}")
        
        # Video writer
        out = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Saving output to: {output_path}")
        
        print(f"\nProcessing video: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}")
        print("Press 'q' to quit, 'p' to pause/resume")
        
        frame_count = 0
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("\nEnd of video")
                        break
                    
                    frame_count += 1
                    
                    # Perform detection
                    results = self.model(frame, verbose=False)
                    
                    # Draw detections
                    annotated_frame = self.draw_detections(frame, results)
                    
                    # Display progress
                    progress = (frame_count / total_frames) * 100
                    progress_text = f"Frame: {frame_count}/{total_frames} ({progress:.1f}%)"
                    detection_count = sum(len(r.boxes) if r.boxes is not None else 0 for r in results)
                    model_type = "Bone Fracture" if self.is_custom_model else "COCO Objects"
                    info_text = f"{progress_text} | Detections: {detection_count} | {model_type}"
                    
                    cv2.putText(
                        annotated_frame,
                        info_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    
                    # Display frame
                    cv2.imshow('Bone Fracture Detection - Video', annotated_frame)
                    
                    # Save frame if requested
                    if save_output and out is not None:
                        out.write(annotated_frame)
                    
                    # Print progress every 30 frames
                    if frame_count % 30 == 0:
                        print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
                
                # Handle keyboard input
                key = cv2.waitKey(1 if not paused else 0) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
            print(f"\nProcessing complete. Processed {frame_count} frames")
            if save_output:
                print(f"Output saved to: {output_path}")
    
    def detect_image(self, image_path, save_output=False, output_path=None):
        """
        Detection on single image
        
        Args:
            image_path: Path to input image
            save_output: Whether to save annotated image
            output_path: Path to save output image (auto-generated if None)
        """
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return
        
        print(f"Processing image: {image_path}")
        
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image: {image_path}")
            return
        
        # Perform detection
        results = self.model(frame, verbose=False)
        
        # Draw detections
        annotated_frame = self.draw_detections(frame, results)
        
        # Count detections
        detection_count = sum(len(r.boxes) if r.boxes is not None else 0 for r in results)
        model_type = "bone fracture" if self.is_custom_model else "COCO object"
        print(f"Found {detection_count} {model_type} detection(s)")
        
        if not self.is_custom_model and detection_count > 0:
            print("⚠ WARNING: This is detecting COCO objects (people, cars, etc.), NOT bone fractures!")
            print("   Train a custom model on your bone fracture dataset to detect fractures.")
        
        # Print detection details
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    if cls in self.model_class_names:
                        class_name = self.model_class_names[cls]
                    elif self.is_custom_model and cls < len(self.CLASS_NAMES):
                        class_name = self.CLASS_NAMES[cls]
                    else:
                        class_name = f"Class {cls}"
                    print(f"  - {class_name}: {conf:.2%} confidence")
        
        # Generate output path if not provided
        if save_output and output_path is None:
            input_path = Path(image_path)
            output_path = str(input_path.parent / f"{input_path.stem}_detected{input_path.suffix}")
        
        # Save if requested
        if save_output:
            cv2.imwrite(output_path, annotated_frame)
            print(f"Saved annotated image to: {output_path}")
        
        # Display image
        cv2.imshow('Bone Fracture Detection - Image', annotated_frame)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Real-time Bone Fracture Detection using YOLO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Webcam detection
  python realtime_yolo_detection.py --source webcam
  
  # Webcam with custom model
  python realtime_yolo_detection.py --source webcam --model path/to/model.pt
  
  # Video file detection
  python realtime_yolo_detection.py --source video.mp4 --save
  
  # Image file detection
  python realtime_yolo_detection.py --source image.jpg --save
  
  # Adjust confidence threshold
  python realtime_yolo_detection.py --source webcam --conf 0.5
        """
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='webcam',
        help='Input source: "webcam", camera ID (e.g., 0, 1), video file path, or image file path'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to custom trained YOLO model (.pt file). If not provided, uses pretrained YOLOv8n'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (0-1). Default: 0.25'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IoU threshold for NMS (0-1). Default: 0.45'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save output video/image'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (auto-generated if not specified)'
    )
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = RealTimeBoneFractureDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Determine input type and process
    source = args.source.lower()
    
    if source == 'webcam' or source.isdigit():
        # Webcam input
        camera_id = int(source) if source.isdigit() else 0
        detector.detect_webcam(
            camera_id=camera_id,
            save_output=args.save,
            output_path=args.output or 'output_webcam.mp4'
        )
    
    elif os.path.isfile(args.source):
        # File input - determine if image or video
        file_ext = Path(args.source).suffix.lower()
        
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
            # Image file
            detector.detect_image(
                image_path=args.source,
                save_output=args.save,
                output_path=args.output
            )
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']:
            # Video file
            detector.detect_video(
                video_path=args.source,
                save_output=args.save,
                output_path=args.output
            )
        else:
            print(f"Error: Unsupported file format: {file_ext}")
    
    else:
        print(f"Error: Invalid source: {args.source}")
        print("Use 'webcam', a camera ID (0, 1, etc.), or a path to an image/video file")


if __name__ == '__main__':
    main()

