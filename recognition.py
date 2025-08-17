import sys
import threading
from functools import lru_cache
import cv2
import numpy as np
import time
import serial
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import postprocess_nanodet_detection

recognition = sys.modules[__name__]

# -------------------- Setup --------------------
threshold = 0.4
iou = 0.65
max_detections = 10
allowed_labels = ["person", "cat", "dog", "horse", "cow", "sheep",
                  "elephant", "bear", "zebra", "giraffe"]
last_detections = []

# -------------------- Serial Communication --------------------
try:
    arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
    time.sleep(2)
    print("✅ Connected to Arduino.")
except:
    arduino = None
    print("⚠️ Failed to connect to Arduino.")

# -------------------- GPIO for Servo --------------------
servo_pin = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)
servo_pwm = GPIO.PWM(servo_pin, 50)
servo_pwm.start(0)

pause_servo = threading.Event()
stop_servo_thread = False

def rotate_servo(angle):
    duty = (angle / 18.0) + 2
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.4)
    servo_pwm.ChangeDutyCycle(0)

def servo_scan():
    angle = 0
    direction = 1
    while not stop_servo_thread:
        if not pause_servo.is_set():
            duty = (angle / 18.0) + 2
            servo_pwm.ChangeDutyCycle(duty)
            time.sleep(0.01)
            angle += direction
            if angle >= 180:
                direction = -1
                angle = 180
            elif angle <= 0:
                direction = 1
                angle = 0

servo_thread = threading.Thread(target=servo_scan)
servo_thread.start()

# -------------------- Detection Class --------------------
class Detection:
    def __init__(self, coords, category, conf, metadata):
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

# -------------------- Detection Parsing --------------------
def parse_detections(metadata):
    global last_detections
    bbox_normalization = intrinsics.bbox_normalization
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_detections

    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = postprocess_nanodet_detection(
            outputs=np_outputs[0], conf=threshold, iou_thres=iou, max_out_dets=max_detections
        )[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h
        boxes = np.array_split(boxes, 4, axis=1)
        boxes = zip(*boxes)

    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]
    return last_detections

@lru_cache
def get_labels():
    labels = intrinsics.labels
    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels

# -------------------- Main --------------------
if __name__ == "__main__":
    model = "./imx500-models-backup/imx500_network_yolov8n_pp.rpk"
    imx500 = IMX500(model)
    intrinsics = imx500.network_intrinsics
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(main={"size": (640, 480)}, buffer_count=6)
    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=False)

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    labels = get_labels()
    recognition.last_command = None

    while True:
        metadata = picam2.capture_metadata()
        detections = parse_detections(metadata)
        frame = picam2.capture_array()

        detected = False
        for detection in detections:
            label_name = labels[int(detection.category)]
            if label_name in allowed_labels:
                detected = True
                x, y, w, h = detection.box
                label = f"{label_name} ({detection.conf:.2f})"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x + 5, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print(f"🟢 Detected: {label}")

        # Control servo scan
        if detected:
            pause_servo.set()
        else:
            pause_servo.clear()

        # Send command to Arduino
        if arduino:
            command = b'stop\n' if detected else b'resume\n'
            if recognition.last_command != command:
                try:
                    arduino.write(command)
                    recognition.last_command = command
                    print("➡️ Sent to Arduino:", command.strip().decode())
                except Exception as e:
                    print("⚠️ Serial error:", e)

        cv2.imshow("Live Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_servo_thread = True
    servo_thread.join()
    picam2.stop()
    cv2.destroyAllWindows()
    if arduino:
        arduino.close()
    GPIO.cleanup()

