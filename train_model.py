import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# === Parameters ===
dataset_path = 'INbreast/CLAHE_images'
img_size = 128
batch_size = 32
initial_epochs = 20
finetune_epochs = 5

os.makedirs("model", exist_ok=True)

# === Data Augmentation ===
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# === Class Weights ===
labels = train_gen.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(class_weights))

# === Model Architecture ===
def build_model():
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    base.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    output = Dense(train_gen.num_classes, activation='softmax')(x)
    return Model(inputs=base.input, outputs=output)

# === Callbacks ===
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6),
    ModelCheckpoint('model/best_model.keras', save_best_only=True)
]

# === Initial Training ===
model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=initial_epochs,
    callbacks=callbacks,
    class_weight=class_weights
)

# === Fine-tuning ===
model = load_model('model/best_model.keras')

for layer in model.layers:
    layer.trainable = True  # Unfreeze all layers

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=finetune_epochs,
    callbacks=callbacks,
    class_weight=class_weights
)

# Save final model
model.save('model/final_finetuned_model.keras')
