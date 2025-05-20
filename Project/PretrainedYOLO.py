from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load a pretrained mode
model = YOLO('yolov8n.pt')

# Train on the dataset
model.train(data='data.yaml', epochs=100, imgsz=640, batch=8, 
            optimizer='AdamW',  # Change optimizer here (SGD, Adam, AdamW, RMSprop)
            lr0=0.01, lrf=0.1,  # Learning rate tuning
            weight_decay=0.0005,  # Regularization to avoid overfitting
            momentum=0.9,  # Helps stabilize training
            flipud=0.5, fliplr=0.5,  # Data augmentation
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # Color augmentation
            patience=20, device="cuda")

# Evaluate the model on test images
model.val(split="test")

#print(metrics)


"""
# Predict 
image_path = "/media/Daten/datasets/Test/images/15_239_101_64_47_jpg.rf.c23cc3f6ba46896ce9c2cf3ee242bbaf.jpg"
results = model.predict(image_path,
              save=True,
              imgsz=640,
              conf=0.3, 
              iou=0.5)

# Extract the first result
result = results[0]

# Load the image using OpenCV
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib

# Plot the image with detections
plt.figure(figsize=(8, 6))
plt.imshow(img)
plt.axis("off")

# Draw bounding boxes
for box in result.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert coordinates to integer
    conf = box.conf[0].item()  # Get confidence score
    cls = int(box.cls[0].item())  # Get class ID
    label = f"{model.names[cls]}: {conf:.2f}"  # Class label and confidence

    # Draw rectangle and label
    plt.text(x1, y1 - 5, label, color="red", fontsize=12, backgroundcolor="white")
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor="red", linewidth=2, fill=False))

# Show the image with detections
plt.show()

print("\n[INFO] Inference completed. Results saved!")
"""