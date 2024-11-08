from flask import Flask, render_template, Response, redirect, url_for
import cv2
import numpy as np
import dlib
from imutils import face_utils
import winsound  # For sound alerts on Windows
from twilio.rest import Client  # For sending WhatsApp messages
app = Flask(__name__, static_url_path='/static', static_folder='static')


# Initialize the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Twilio credentials (use your actual credentials)
account_sid = 'ACd495ea7ed32dbacbb5a31edb62785695'
auth_token = 'f587bd9a464147784848af4e32a6249c'
twilio_client = Client(account_sid, auth_token)

def send_whatsapp_alert(message):
    """Send a WhatsApp alert using Twilio."""
    twilio_client.messages.create(
        from_='whatsapp:+14155238886',  # Twilio sandbox number
        body=message,
        to='whatsapp:+919459135009'
    )

def compute(ptA, ptB):
    """Compute the Euclidean distance between two points."""
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(a, b, c, d, e, f):
    """Check if an eye is closed based on the ratio of distances."""
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2  # Eyes open
    elif 0.21 < ratio <= 0.25:
        return 1  # Eyes semi-open (drowsy)
    else:
        return 0  # Eyes closed (sleeping)

def generate_frames():
    # Initialize the camera
    cap = cv2.VideoCapture(0)

    # Check if the camera opened correctly
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    sleep, drowsy, active = 0, 0, 0
    status = ""
    color = (0, 0, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], 
                                 landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], 
                                  landmarks[47], landmarks[46], landmarks[45])

            if left_blink == 0 or right_blink == 0:
                sleep += 1
                drowsy = 0
                active = 0
                if sleep > 6:
                    status = "WARNING DRIVER IS SLEEPING !!!"
                    color = (0, 0, 255)
                    winsound.Beep(1000, 1000)
                    send_whatsapp_alert("Alert: Driver is sleeping! Immediate attention needed!")

            elif left_blink == 1 or right_blink == 1:
                sleep = 0
                active = 0
                drowsy += 1
                if drowsy > 6:
                    status = "Drowsy!"
                    color = (0, 255, 255)
                    winsound.Beep(1000, 500)
                    send_whatsapp_alert("Alert: Driver is drowsy. Stay cautious!")

            else:
                sleep = 0
                drowsy = 0
                active += 1
                if active > 6:
                    status = "Driver is Active"
                    color = (0, 255, 0)

            cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')  # Serve the HTML page

@app.route('/start')
def start_camera():
    # Redirect to the video feed endpoint when the button is clicked
    return redirect(url_for('video_feed'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
