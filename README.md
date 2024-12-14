# **Cat vs Dog Classifier**

โปรเจคนี้เป็นการพัฒนาโมเดลปัญญาประดิษฐ์ (AI) สำหรับแยกแยะภาพระหว่าง "แมว" และ "หมา" โดยใช้เทคโนโลยีการเรียนรู้ของเครื่อง (Machine Learning) และการเรียนรู้เชิงลึก (Deep Learning) ด้วย TensorFlow และ Keras

---

## **ฟีเจอร์หลัก**
- แยกแยะภาพระหว่างแมวและหมา (Cat vs Dog) ด้วยความแม่นยำสูง
- ใช้ Data Augmentation เพื่อเพิ่มความหลากหลายของข้อมูล
- แสดงผลลัพธ์การเทรนผ่านกราฟ Accuracy และ Loss (รองรับภาษาไทย)
- ทดสอบโมเดลด้วยข้อมูลใหม่จากกล้อง (Real-time Testing)

---

## **เทคโนโลยีที่ใช้**
- **ภาษาโปรแกรม**: Python 3.10+
- **ไลบรารีหลัก**:
  - TensorFlow และ Keras
  - OpenCV (สำหรับการทดสอบด้วยกล้อง)
  - Matplotlib (สำหรับการแสดงกราฟผลลัพธ์)
  - Scikit-learn (สำหรับการแบ่งข้อมูล Train/Test)

---

## **โครงสร้างโปรเจค**
```plaintext
ML_DOGCAT/
├── dataset/
│   ├── cat/                  # รูปภาพแมวสำหรับการเทรน
│   ├── dog/                  # รูปภาพหมาสำหรับการเทรน
├── prepared_dataset/         # ข้อมูลที่แบ่งเป็น Train/Validation/Test
│   ├── train/
│   ├── validation/
│   ├── test/
├── models/
│   ├── cat_dog_model.h5      # โมเดลที่เทรนแล้ว
├── src/                      # โค้ดหลักของโปรเจค
│   ├── main.py               # จุดเริ่มต้นของโปรแกรม
│   ├── data_loader.py        # โหลดและจัดการข้อมูล
│   ├── train_model.py        # สร้างและเทรนโมเดล
│   ├── evaluate_model.py     # ประเมินและแสดงผลโมเดล
├── venv/                     # Virtual Environment
├── README.md                 # เอกสารอธิบายโปรเจค
