import cv2
import zmq
import pickle
import time

# ------------------ ZMQ SETUP ------------------
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.setsockopt(zmq.SNDHWM, 1)
socket.bind("tcp://*:5555")

# ------------------ CAMERA ------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera not accessible")
    exit(1)

print("📷 Camera service started on port 5555")

# Give subscribers time to connect
time.sleep(0.02)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    cv2.putText(frame, "SHARED CAMERA", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    data = pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL)
    socket.send(data)
