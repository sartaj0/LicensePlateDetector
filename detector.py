import numpy as np
import time
import cv2
import os
from imutils.video import FPS
import imutils

# from easyocr import Reader

def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()

def image_to_text(image):
    reader = Reader(['en'], gpu=False)
    results = reader.readtext(image)
    text=""
    for (bbox, text, prob) in results:
    	# display the OCR'd text and associated probability
    	print("[INFO] {:.4f}: {}".format(prob, text))
    	# unpack the bounding box
    	(tl, tr, br, bl) = bbox
    	tl = (int(tl[0]), int(tl[1]))
    	tr = (int(tr[0]), int(tr[1]))
    	br = (int(br[0]), int(br[1]))
    	bl = (int(bl[0]), int(bl[1]))
    	# cleanup the text and draw the box surrounding the text along
    	# with the OCR'd text itself
    	text = cleanup_text(text)
    return text
minConfi = 0.1
thresh = 0.3

labelPaths = "obj.names"
Labels = open(labelPaths).read().strip().split("\n")

np.random.seed(20)

colors = np.random.randint(0, 255, size=(len(Labels), 3),
	dtype="uint8")
# generate random color
#print(colors)

weightsPath = "yolov3_tiny_custom_final.weights"
configPath = "yolov3_tiny_custom.cfg"

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print(ln)
type_file = input("Image or Video: ")

if (type_file == "Video") or (type_file == "video"):
    cap = cv2.VideoCapture('video/dashcam_boston.mp4')

    ret, image = cap.read()
    image = imutils.resize(image, height=800)
    (H, W) = image.shape[:2]
    writer = cv2.VideoWriter('output1.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 30, (W, H))
    fps = FPS().start()
    while True:
        ret, image = cap.read()
        if image is None:
            break
        # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = imutils.resize(image, height=600)
        

        (H, W) = image.shape[:2]

        # determine only the *output* layer names that we need from YOLO


        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)

        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # print("[INFO] YOLO took {:.6f} seconds".format(end - start))

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > minConfi:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    #print(classIDs)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, minConfi, thresh)


        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                
                color = [int(c) for c in colors[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
                
                text = "{}: {:.4f}".format(Labels[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                #print(text)
        fps.update()
        fps.stop()
        cv2.putText(image, "FPS{:.2f}".format(fps.fps()), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Image", image)
        
        cv2.waitKey(1)
        writer.write(image)
    writer.release()
    cap.release()
    cv2.destroyAllWindows()
elif (type_file == "Image") or (type_file == "image"):
    images = os.listdir("image")
    for img in images:
        image = cv2.imread(os.path.join("image", img))
        
        (H, W) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)

        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > minConfi:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, minConfi, thresh)


        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                
                color = [int(c) for c in colors[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                # text = image_to_text(image[y:y+h, x:x+w])
                # print(text)
                text = "{}: {:.4f}".format(Labels[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        image = imutils.resize(image, width=600)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
