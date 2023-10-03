import os
import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw
import subprocess
import time  # Add the time module

model = YOLO("C:\\Users\\jaikr\\Downloads\\Final\\best.pt")

# Folder containing input images
input_folder = "C:\\Users\\jaikr\\Downloads\\Final\\train\\images"
output_folder = "output_images"  # Output folder for saving images
output_video = "output_video.mp4"  # Output video file name

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get a list of image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith((".jpg", ".jpeg", ".png"))]

# Sort the image files to maintain order
image_files.sort()

for idx, image_file in enumerate(image_files):
    image_path = os.path.join(input_folder, image_file)

    try:
        results = model.predict(source=image_path)

        for r in results:
            print(r.keypoints)

            # This line is changed
            keypoints = r.keypoints.xy.int().numpy()  # Get the keypoints
            img_array = r.plot(kpt_line=True, kpt_radius=6)  # Plot a BGR array of predictions
            im = Image.fromarray(img_array[..., ::-1])  # Convert array to a PIL Image

            draw = ImageDraw.Draw(im)
            draw.line([(keypoints[0][0][0], keypoints[0][0][1]), (keypoints[0][1][0],
                                                                  keypoints[0][1][1]),
                       (keypoints[0][2][0], keypoints[0][2][1])], fill=(255, 0, 0), width=5)

            # Save the image with a sequence number
            output_path = os.path.join(output_folder, f"output_image_{idx:04d}.png")
            im.save(output_path)

        print(f"Processed image '{image_file}'.")

    except Exception as e:
        print(f"Error processing image '{image_file}': {e}")
        continue  # Continue to the next image if an error occurs

print("Image processing completed.")

# Use OpenCV to create a video from the saved images with a delay of 0.5 seconds
frame_array = []
for i in range(len(image_files)):
    img_path = os.path.join(output_folder, f"output_image_{i:04d}.png")

    # Check if the image file exists
    if not os.path.exists(img_path):
        print(f"Image '{img_path}' not found or has an issue with format. Skipping.")
        continue

    img = cv2.imread(img_path)
    height, width, layers = img.shape
    size = (width, height)
    frame_array.append(img)

out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

for i in range(len(frame_array)):
    out.write(frame_array[i])
    time.sleep(1)  # Add a delay of 0.5 seconds between frames

out.release()

print(f"Video '{output_video}' created successfully.")


