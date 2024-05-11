import numpy as np
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
import matplotlib.pyplot as plt
import os
import cv2
import random
import sklearn.model_selection as model_selection
import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from contextlib import redirect_stdout
#categories = ['actinic_keratosis','basal_cell_carcinoma','dermatofibroma','melanoma']
dir = "data/"
categories = [folder for folder in os.listdir(dir) if os.path.isdir(os.path.join(dir, folder))]
print(categories)
SIZE = 24
def getData():
    rawdata = []
    data = []
    dir = "data/"
    for category in categories:
        path = os.path.join(dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                rawdata = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_data = cv2.resize(rawdata, (SIZE, SIZE))

                data.append([new_data, class_num])
            except Exception as e:
                pass

    random.shuffle(data)

    img_data = []
    img_labels = []
    for features, label in data:
        img_data.append(features)
        img_labels.append(label)
    img_data = np.array(img_data).reshape(-1, SIZE, SIZE, 1)
    img_data = img_data / 255.0
    img_labels = np.array(img_labels)

    return img_data, img_labels




def createModel(train_data=None):
    if os.path.exists('./model/skin.h5') and train_data is None:
        try:
            print(__name__)
            model = keras.models.load_model('./model/skin.h5')
            print("returned")
            return model
        except Exception as e:
            print("error")


    elif train_data is not None:
        model = keras.Sequential([

            keras.Input(shape=train_data.shape[1:]),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),

            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax")
        ])
        return model
    

def train_model_for_disease(disease_name):
    # Define data directories
    train_dir = os.path.join('data', disease_name, 'train')
    val_dir = os.path.join('data', disease_name, 'val')

    # Define image size and batch size
    img_size = (224, 224)
    batch_size = 32

    # Define data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Load pre-trained MobileNetV2 model
    base_model = MobileNetV2(weights='imagenet', include_top=False)

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_generator, epochs=10, validation_data=val_generator)

    # Save the trained model
    model.save(f'models/{disease_name}_model.h5')


if __name__ == "__main__":
    # Example usage
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--disease", required=True, help="Name of the disease to train the model for")
   # args = parser.parse_args()
   # disease_name = args.disease
   # train_model_for_disease('acne')

    data, labels = getData()
    train_data, test_data, train_labels, test_labels = model_selection.train_test_split(data, labels, test_size=0.20)

    train_data, val_data, train_labels, val_labels = model_selection.train_test_split(train_data, train_labels,test_size=0.10)
    print(len(train_data), " ", len(train_labels), len(test_data), " ", len(test_labels))
    num_classes = len(np.unique(labels))
    print(labels)
    model = createModel(train_data)
    checkpoint_path = './model/Sequential_{timestamp}.h5'.format(timestamp=datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')

    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"], )


    history = model.fit(train_data, train_labels, epochs=90
                    )
    model.save('./model/Sequential_{timestamp}.h5'.format(timestamp=datetime.datetime.now().strftime('%Y%m%d%H%M%S')))




#checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')

#test_loss, test_acc = model.evaluate(test_data, test_labels)
#print("Model Accuracy: ", test_acc, "Model Loss: ", test_loss)
#model.summary()
#keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
#weight =model.weights
#print(weight)Check Data Loading:
#print(len(model.weights))

