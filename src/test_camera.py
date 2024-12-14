import cv2
import numpy as np
from tensorflow.keras.models import load_model

# โหลดโมเดลที่เทรนไว้
model = load_model('models/cat_dog_model.h5')  # ใส่ path ของโมเดล

# ขนาดภาพที่โมเดลรองรับ
IMG_SIZE = (150, 150)

# โหลด Haar Cascade สำหรับตรวจจับใบหน้า (หรือใช้ไฟล์ cascade อื่นสำหรับสัตว์)
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(cascade_path)

# เปิดกล้อง
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot access camera")
    exit()

while True:
    # อ่านภาพจากกล้อง
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # แปลงภาพเป็นขาวดำสำหรับการตรวจจับวัตถุ
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตรวจจับวัตถุ (เช่น ใบหน้า)
    objects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # วนลูปผ่านวัตถุที่ตรวจจับได้
    for (x, y, w, h) in objects:
        # ตัดส่วนของภาพตามกรอบ (Bounding Box)
        roi = frame[y:y+h, x:x+w]

        # แปลงภาพสำหรับโมเดล
        resized_roi = cv2.resize(roi, IMG_SIZE)
        normalized_roi = resized_roi / 255.0
        reshaped_roi = np.expand_dims(normalized_roi, axis=0)

        # ใช้โมเดลพยากรณ์
        prediction = model.predict(reshaped_roi)
        class_label = "Dog" if prediction[0][0] > 0.5 else "Cat"
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

        # เพิ่มกรอบสี่เหลี่ยมและข้อความในเฟรม
        label = f"{class_label} ({confidence*100:.2f}%)"
        color = (0, 255, 0) if class_label == "Cat" else (0, 0, 255)  # สีกรอบ: เขียวสำหรับแมว แดงสำหรับหมา
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # แสดงผลลัพธ์
    cv2.imshow("Cat vs Dog Detection", frame)

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและปล่อยทรัพยากร
cap.release()
cv2.destroyAllWindows()
