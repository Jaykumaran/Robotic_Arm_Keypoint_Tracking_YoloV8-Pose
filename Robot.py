import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Create a new YOLOv8n-OBB model from scratch
model = YOLO("C:\\Users\\jaikr\\Downloads\Final\\best.pt")

# Train the model on the DOTAv2 dataset
results = model.predict(source="C:\\Users\\jaikr\\Downloads\\Subset640\\train\\images\\WIN_20230915_20_38_36_Pro.jpg")

# Show the results
for r in results:
    print(r.keypoints)
    img_array = r.plot(kpt_line=True, kpt_radius=3)  # plot a BGR numpy array of predictions

    # Extract xy coordinates for the keypoints
    keypoints = r.xy.int().numpy()

    # Draw lines between the keypoints
    cv2.line(img_array, tuple(keypoints[0]), tuple(keypoints[1]), (255, 0, 255), 2)  # line from point1 to point2
    cv2.line(img_array, tuple(keypoints[1]), tuple(keypoints[2]), (255, 0, 255), 2)  # line from point2 to point3

    img = Image.fromarray(img_array[..., ::-1])  # create a PIL image from the array
    img.show() # show the image
