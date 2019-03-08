import numpy as np
import cv2

def objectDetection():
    classes = ["Background", "Airplane", "Bike", "Bird", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cow",
               "Table", "Dog", "Horse", "Motorbike", "Person", "Plant", "Sheep", "Sofa", "Train", "TV / Monitor"]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    net = cv2.dnn.readNetFromCaffe("model/MobileNetSSD_deploy.prototxt.txt", 'model/MobileNetSSD_deploy.caffemodel')

    video_capture = cv2.VideoCapture(0)
    print('Opening video feed...(Press q to close)')

    while True:
        # if the `q` is pressed, break from the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Closing video feed...')
            break
        # Capture frame-by-frame
        # ret: signals if you have run out of frames; for video file use only
        # frame: frame by frame video input
        ret, frame = video_capture.read()
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and predictions
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            # extract the confidence associated with the prediction and filter out weak detectio
            confidence = detections[0, 0, i, 2]
            if confidence > 0.35:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                index = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
                label = "{} ({:.2f}%)".format(classes[index],confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),colors[index], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[index], 1, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Object Detection', frame)

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

objectDetection()
