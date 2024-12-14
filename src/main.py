import sys
import os

# เพิ่ม path ของโฟลเดอร์ parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data
from src.train_model import create_model, train_model
from src.evaluate_model import evaluate_model, plot_training

# โหลดข้อมูล
train_data, val_data = load_data(data_dir='dataset/train')

# สร้างและเทรนโมเดล
model = create_model()
model, history = train_model(model, train_data, val_data, epochs=10)

# บันทึกโมเดล
model.save('models/cat_dog_model.h5')

# ประเมินผล
test_data, _ = load_data(data_dir='dataset/test', img_size=(150, 150), batch_size=32)
evaluate_model(model, test_data)

# แสดงกราฟ
plot_training(history)
