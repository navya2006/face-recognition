import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
import face_model

app = Flask(__name__)

camera = cv2.VideoCapture(0)

mode = "idle"

def gen_frames():

    global mode

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    recognizer = None
    labels = None

    while True:

        success, frame = camera.read()

        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        if mode == "recognize":

            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read("trainer.yml")

            labels = np.load("labels.npy", allow_pickle=True).item()

        for (x, y, w, h) in faces:

            name = "Face"

            if mode == "capture":
                name = "Capturing..."

            elif mode == "recognize" and recognizer is not None:

                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

                if confidence < 80:
                    name = labels[id]
                else:
                    name = "Unknown"

            # draw box
            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            # draw name
            cv2.putText(
                frame,
                name,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/")
def home():
    return render_template("index.html")



@app.route("/register", methods=["POST"])
def register():

    global mode

    name = request.json["name"]
    user_id = request.json["id"]

    mode = "capture"

    face_model.createUser(user_id, name)

    face_model.train()

    mode = "idle"

    return jsonify({"message": "Face captured and trained"})



@app.route("/recognize")
def recognize():

    global mode

    mode = "recognize"

    person = face_model.recognize()

    return jsonify({"person": person})


if __name__ == "__main__":
    app.run(debug=True)