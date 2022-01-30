import cv2


# importing the trained data from openCv library
face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# caputuring the camera(connected device)
# img = cv2.imread(' ')
cam = cv2.VideoCapture(0)


while True:
    successful_frame_read, frame = cam.read()

    # greyscale covention
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = face_data.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 256, 0), 10)

    # displaying all faces

    cv2.imshow("Facial recognition software", frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

# break the web connection
cam.release()
