import cv2
import numpy as np
from transformers import AutoImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import time

# Load face detector model
print("[INFO] loading face detector...")
protoPath = "models/face_detector/deploy.prototxt"
modelPath = "models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load ViT model and processor for gender classification
print("[INFO] loading gender classifier...")
image_processor = AutoImageProcessor.from_pretrained("models/gender/rizvandwiki")
model = ViTForImageClassification.from_pretrained("models/gender/rizvandwiki")
model.eval()

# Initialize video capture
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(3, 640)
cap.set(4, 480)

min_size = 100  # Minimum face size to process
confidence_threshold = 0.8  # Confidence threshold for face detection

while True:
    try:
        # Record start time for FPS calculation
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            continue

        # Resize frame for consistent processing
        frame = cv2.resize(frame, (640, 480))
        (h, w) = frame.shape[:2]

        # Prepare blob for face detection
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        detections = faceNet.forward()

        if detections.shape[2] <= 0:
            continue

        # Loop over face detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter weak detections
            if confidence > confidence_threshold:
                # Compute bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Ensure bounding box is within frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                # Skip small faces
                if abs(endX - startX) < min_size or abs(endY - startY) < min_size:
                    continue

                # Extract face ROI for gender classification
                face = frame[startY:endY, startX:endX]
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)

                # Process face for ViT model
                inputs = image_processor(face_pil, return_tensors="pt")
                
                # Perform gender prediction
                with torch.no_grad():
                    logits = model(**inputs).logits
                predicted_label = logits.argmax(-1).item()
                gender = model.config.id2label[predicted_label]
                confidence_score = torch.softmax(logits, dim=-1)[0][predicted_label].item()

                # Draw bounding box and gender label
                color = (0, 255, 0) if gender.lower() == 'male' else (0, 0, 255)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                label = f"{gender}: {confidence_score:.2f}"
                cv2.putText(frame, label, (startX, startY - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # Calculate FPS
        end_time = time.time()
        processing_time = end_time - start_time
        fps = 1 / processing_time if processing_time > 0 else 0

        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow('Face and Gender Detection', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")

# Cleanup
cap.release()
cv2.destroyAllWindows()