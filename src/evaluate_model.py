import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', family='Tahoma')  # หากไม่มี Tahoma ให้เปลี่ยนเป็นฟอนต์อื่นที่มีในระบบ

def evaluate_model(model, test_data):
    # ประเมินผลโมเดลด้วยข้อมูลชุดทดสอบ
    results = model.evaluate(test_data, verbose=1)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    return results

def plot_training(history):
    # สร้างกราฟ Accuracy และ Loss
    plt.figure(figsize=(12, 6))

    #(Accuracy)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('กราฟความแม่นยำ (Accuracy)')
    plt.xlabel('รอบการเทรน (Epoch)')
    plt.ylabel('ความแม่นยำ (Accuracy)')
    plt.legend()
    plt.grid()

    #(Loss)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('กราฟค่าเสียหาย (Loss)')
    plt.xlabel('รอบการเทรน (Epoch)')
    plt.ylabel('ค่าเสียหาย (Loss)')
    plt.legend()
    plt.grid()

    # แสดงกราฟ
    plt.tight_layout()
    plt.show()
