from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir, img_size=(150, 150), batch_size=32):
    # Data Augmentation สำหรับ Train
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # แบ่งข้อมูล Validation ออกจาก Train
    )

    # สร้าง Train Generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    # สร้าง Validation Generator
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    print(f"Found {train_generator.samples} training samples.")
    print(f"Found {validation_generator.samples} validation samples.")
    return train_generator, validation_generator
