
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from PIL import Image

# Load YOLO
net = cv2.dnn.readNet("/content/gdrive/MyDrive/v3_car/darknet/backup/yolov3_final.weights",
                      "/content/gdrive/MyDrive/v3_car/darknet/yolov3.cfg")
layer_names = net.getLayerNames()
unconnected_out_layers = net.getUnconnectedOutLayers().flatten()

# Debugging print to check layer names and unconnected out layers
print("Layer names: ", layer_names)
print("Unconnected out layers: ", unconnected_out_layers)

# Ensure we handle the returned values correctly
output_layers = [layer_names[i - 1] for i in unconnected_out_layers]

# Load classes
classes = []
with open("/content/gdrive/MyDrive/v3_car/darknet/data/classes.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def detect_and_count(image):
    height, width, channels = image.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(image, 0.00392, (608, 608), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.25 and classes[class_id] == "car":
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

    # Count detected cars
    count = len(indexes)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
            cv2.putText(image, f'Count: {count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return image, count

# Load video
cap = cv2.VideoCapture('/content/gdrive/MyDrive/Driving Downtown - New York City 4K - USA.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    frame, count = detect_and_count(frame)
    cv2_imshow(frame)  # Use cv2_imshow instead of cv2.imshow

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
