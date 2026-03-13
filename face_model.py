import cv2
import numpy as np
import os

def createUser(id, name):

    cam = cv2.VideoCapture(0)

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    count = 0

    os.makedirs(f"dataset/{name}", exist_ok=True)

    while True:

        ret, img = cam.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:

            count += 1

            face_img = gray[y:y+h, x:x+w]

            cv2.imwrite(
                f"dataset/{name}/{count}.jpg",
                face_img
            )

        if count >= 50:
            print("Done Capturing Images")
            break

    cam.release()


def train():

    import os
    import cv2
    import numpy as np

    dataset_path = "dataset"

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []

    label_map = {}
    label_id = 0

    for person in os.listdir(dataset_path):

        person_path = os.path.join(dataset_path, person)

        # skip hidden files like .DS_Store
        if not os.path.isdir(person_path):
            continue

        label_map[label_id] = person

        for img_name in os.listdir(person_path):

            img_path = os.path.join(person_path, img_name)

            # skip non images
            if not img_name.endswith(".jpg"):
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            faces.append(img)
            labels.append(label_id)

        label_id += 1

    print("Faces collected:", len(faces))

    if len(faces) == 0:
        print("No faces found for training")
        return

    recognizer.train(faces, np.array(labels))

    recognizer.save("trainer.yml")

    np.save("labels.npy", label_map)

    print("Model trained successfully")


def recognize():

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.read("trainer.yml")

    labels = np.load("labels.npy", allow_pickle=True).item()

    cam = cv2.VideoCapture(0)

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    name = "Unknown"

    while True:

        ret, img = cam.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:

            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 80:
                name = labels[id]

        break

    cam.release()

    return name