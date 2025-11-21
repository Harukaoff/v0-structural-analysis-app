"""
YOLO Object Detection Script for Structural Elements
Detects beams, supports, loads, and moments from hand-drawn structural diagrams
"""

import sys
import json
import base64
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image

# Note: This script requires ultralytics to be installed
try:
    from ultralytics import YOLO
except ImportError:
    print(json.dumps({"error": "ultralytics not installed. Run: pip install ultralytics"}))
    sys.exit(1)

# MODEL_PATH is now optional and can be overridden
MODEL_PATH = Path(__file__).parent.parent / "models" / "best.pt"

# Class names mapping
CLASS_NAMES = {
    0: "beam",
    1: "pin",
    2: "roller",
    3: "fixed",
    4: "hinge",
    5: "load",
    6: "UDL",
    7: "momentL",
    8: "momentR"
}

_model_cache = {}

def load_model(model_path=None):
    """Load YOLO model with caching"""
    print(f"[v0] load_model called with path: {model_path}", file=sys.stderr)
    
    if model_path is None:
        model_path = MODEL_PATH
    
    if isinstance(model_path, Path):
        model_path = str(model_path)
    
    print(f"[v0] Checking model path: {model_path}", file=sys.stderr)
    print(f"[v0] Model exists: {Path(model_path).exists()}", file=sys.stderr)
    
    if model_path is None or not Path(model_path).exists():
        print(f"[v0] Model not found at: {model_path}", file=sys.stderr)
        return None
    
    # Check cache
    if model_path in _model_cache:
        print(f"[v0] Using cached model", file=sys.stderr)
        return _model_cache[model_path]
    
    try:
        print(f"[v0] Loading YOLO model from: {model_path}", file=sys.stderr)
        model = YOLO(model_path)
        _model_cache[model_path] = model
        print(f"[v0] Model loaded successfully", file=sys.stderr)
        return model
    except Exception as e:
        print(f"[v0] Error loading model: {e}", file=sys.stderr)
        import traceback
        print(f"[v0] Traceback:\n{traceback.format_exc()}", file=sys.stderr)
        return None

def decode_base64_image(base64_string):
    """Decode base64 image string to PIL Image"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image

def detect_elements(image_base64, model_path=None, conf_threshold=0.25):
    """
    Detect structural elements in the image
    Returns detected elements with bounding boxes and classifications
    """
    print(f"[v0] detect_elements called", file=sys.stderr)
    print(f"[v0] Model path: {model_path}", file=sys.stderr)
    print(f"[v0] Confidence threshold: {conf_threshold}", file=sys.stderr)
    
    model = load_model(model_path)
    if model is None:
        error_msg = "Failed to load YOLO model. Please upload a model file."
        print(f"[v0] {error_msg}", file=sys.stderr)
        return {"error": error_msg}
    
    try:
        # Decode image
        print(f"[v0] Decoding image...", file=sys.stderr)
        image = decode_base64_image(image_base64)
        print(f"[v0] Image decoded: {image.width}x{image.height}", file=sys.stderr)
        
        # Run YOLO detection
        print(f"[v0] Running YOLO detection...", file=sys.stderr)
        results = model(image, conf=conf_threshold)
        print(f"[v0] Detection complete, processing results...", file=sys.stderr)
        
        detected_elements = []
        
        for result in results:
            if hasattr(result, 'obb') and result.obb is not None:
                # Oriented bounding boxes
                print(f"[v0] Processing OBB detections...", file=sys.stderr)
                boxes = result.obb.xyxyxyxy.cpu().numpy()
                classes = result.obb.cls.cpu().numpy()
                confidences = result.obb.conf.cpu().numpy()
                
                print(f"[v0] Found {len(boxes)} OBB detections", file=sys.stderr)
                
                for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
                    element_type = CLASS_NAMES.get(int(cls), "unknown")
                    
                    # Calculate center point and angle
                    points = box.reshape(-1, 2)
                    center_x = float(np.mean(points[:, 0]))
                    center_y = float(np.mean(points[:, 1]))
                    
                    # Calculate angle from oriented box
                    dx = points[1][0] - points[0][0]
                    dy = points[1][1] - points[0][1]
                    angle = float(np.degrees(np.arctan2(dy, dx)))
                    
                    detected_elements.append({
                        "id": i,
                        "type": element_type,
                        "confidence": float(conf),
                        "bbox": box.tolist(),
                        "center": {"x": center_x, "y": center_y},
                        "angle": angle,
                        "width": float(np.linalg.norm(points[1] - points[0])),
                        "height": float(np.linalg.norm(points[2] - points[1]))
                    })
            elif hasattr(result, 'boxes') and result.boxes is not None:
                # Regular bounding boxes
                print(f"[v0] Processing regular box detections...", file=sys.stderr)
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                print(f"[v0] Found {len(boxes)} box detections", file=sys.stderr)
                
                for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
                    element_type = CLASS_NAMES.get(int(cls), "unknown")
                    
                    x1, y1, x2, y2 = box
                    center_x = float((x1 + x2) / 2)
                    center_y = float((y1 + y2) / 2)
                    
                    detected_elements.append({
                        "id": i,
                        "type": element_type,
                        "confidence": float(conf),
                        "bbox": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        "center": {"x": center_x, "y": center_y},
                        "angle": 0,
                        "width": float(x2 - x1),
                        "height": float(y2 - y1)
                    })
        
        print(f"[v0] Total elements detected: {len(detected_elements)}", file=sys.stderr)
        
        return {
            "success": True,
            "image_width": image.width,
            "image_height": image.height,
            "elements": detected_elements,
            "counts": {
                "beam": sum(1 for e in detected_elements if e["type"] == "beam"),
                "supports": sum(1 for e in detected_elements if e["type"] in ["pin", "roller", "fixed", "hinge"]),
                "loads": sum(1 for e in detected_elements if e["type"] in ["load", "UDL", "momentL", "momentR"])
            }
        }
        
    except Exception as e:
        error_msg = f"Detection failed: {str(e)}"
        print(f"[v0] {error_msg}", file=sys.stderr)
        import traceback
        print(f"[v0] Traceback:\n{traceback.format_exc()}", file=sys.stderr)
        return {"error": error_msg}

if __name__ == "__main__":
    # Read base64 image from stdin
    input_data = sys.stdin.read()
    data = json.loads(input_data)
    
    image_base64 = data.get("image")
    if not image_base64:
        print(json.dumps({"error": "No image provided"}))
        sys.exit(1)
    
    # Detect elements
    result = detect_elements(image_base64)
    
    # Output result as JSON
    print(json.dumps(result))
