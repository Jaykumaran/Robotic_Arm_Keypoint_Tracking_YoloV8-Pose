import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image, ImageDraw

model = YOLO("C:\\Users\\jaikr\\Downloads\\Final\\best.pt")
results = model.predict(source="C:\\Users\\jaikr\\Downloads\\Subset640\\train\\images\\WIN_20230915_20_27_08_Pro.jpg")

for r in results:
    print(r.keypoints)

    # this line is changed
    keypoints = r.keypoints.xy.int().numpy()  # get the keypoints
    img_array = r.plot(kpt_line=True, kpt_radius=6)  # plot a BGR array of predictions
    im = Image.fromarray(img_array[..., ::-1])  # Convert array to a PIL Image

    draw = ImageDraw.Draw(im)
    draw.line([(keypoints[0][0][0], keypoints[0][0][1]), (keypoints[0][1][0],
            keypoints[0][1][1]), (keypoints[0][2][0], keypoints[0][2][1])], fill=(0, 0,255), width=5)
    im.show()
