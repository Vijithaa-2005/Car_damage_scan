import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

train_dir = r"C:\pro_final\Car-Damage-Assessment-AI\data\archive (5)\data3a\training"
val_dir = r"C:\pro_final\Car-Damage-Assessment-AI\data\archive (5)\data3a\validation"

datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Automatically detect number of classes
num_classes = len(train_data.class_indices)
print("Number of classes detected:", num_classes)
print("Class mapping:", train_data.class_indices)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')   # <-- Automatic output layer
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_data, validation_data=val_data, epochs=10)

model.save("vehicle_damage_vijithaa.h5")
print("✅ Model training completed & saved")
