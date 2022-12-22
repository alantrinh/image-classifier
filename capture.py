import sys
import logging
import os
import cv2
from utils import write_image, key_action, init_cam

import numpy as np
import pandas as pd
import keras_tuner as kt
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy

def process_image(image_path):
    image = load_img(image_path, target_size=(224,224))
    image_array = img_to_array(image)
    image_batch = np.expand_dims(image_array, axis=0)
    processed_image = preprocess_input(image_batch)
    return processed_image

def calculate_image_class_probabilities(model, preprocessed_image):
    base_path = '../data/train'
    CLASSES = [f for f in os.listdir(base_path) if not f.startswith('.')]
    probabilities = model.predict(
        preprocessed_image,
        verbose=0
    )
    probabilities = np.round(probabilities,6)[0].tolist()
    class_probabilities = {
        k:v for (k,v) in zip(CLASSES, probabilities)
    }
    return class_probabilities

def train_model():
    base_path = '../data/train'
    CLASSES = [f for f in os.listdir(base_path) if not f.startswith('.')]

    BASE_MODEL = MobileNetV2(
        weights='imagenet',
        alpha=1.0,
        include_top=False,
        pooling='avg',
        input_shape=(224,224,3)
    )

    BASE_MODEL.trainable = False

    def build_hypermodel(hyperparameters):
        model = Sequential()
        model.add(BASE_MODEL)
        # tune number of neurons in first dense layer
        units = hyperparameters.Choice(
            'units', 
            values=[1280,640,320,160,80],
        )
        # first dense layer
        model.add(
            Dense(
                units=units, 
                activation='relu'
            )
        )
        # second layer (dropout layer) 
        model.add(
            Dropout(
                rate=0.5
            )
        )
        # output layer with softmax activation function
        model.add(
            Dense(
                len(CLASSES),
                activation='softmax'
            )
        )
        # tune learning rate for the optimizer
        learning_rate = hyperparameters.Choice(
            'learning_rate', 
            values=[1e-2, 1e-3, 1e-4]
        )
        # compile model
        model.compile(
            optimizer=Adam(
                learning_rate=learning_rate
            ),
            loss=categorical_crossentropy,
            metrics=[categorical_accuracy]
        )

        return model

    image_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_image_data = image_data_generator.flow_from_directory(
        directory=base_path,
        class_mode='categorical',
        classes=CLASSES,
        batch_size=400,
        target_size=(224, 224)
    )

    Xtrain, ytrain = next(train_image_data)

    tuner = kt.Hyperband(
        hypermodel=build_hypermodel,
        objective='val_categorical_accuracy',
        max_epochs=50,
        hyperband_iterations=3,
        project_name='data/kt_files'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss', 
        patience=5
    )

    tuner.search(Xtrain, ytrain,epochs=100, callbacks=[early_stop_callback], validation_split=0.2)

    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    best_hypermodel = tuner.hypermodel.build(best_hyperparameters)

    best_hypermodel.fit(Xtrain, ytrain,
        epochs=50,
        batch_size=20,
        callbacks=[early_stop_callback],
        validation_split=0.2
    )

    print('\n\n###HYPERMODEL TRAINING DONE. RETURNING SELECTED MODEL###\n\n')
    return best_hypermodel

if __name__ == "__main__":

    trained_model = train_model()

    # folder to write images to
    out_folder = sys.argv[1]

    # maybe you need this
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    logging.getLogger().setLevel(logging.INFO)
   
    # also try out this resolution: 640 x 360
    webcam = init_cam(640, 480)
    key = None

    try:
        # q key not pressed 
        while key != 'q':
            # Capture frame-by-frame
            ret, frame = webcam.read()
            # fliping the image 
            frame = cv2.flip(frame, 1)
   
            # draw a [224x224] rectangle into the frame, leave some space for the black border 
            offset = 2
            width = 224
            x = 160
            y = 120
            cv2.rectangle(img=frame, 
                          pt1=(x-offset,y-offset), 
                          pt2=(x+width+offset, y+width+offset), 
                          color=(0, 0, 0), 
                          thickness=2
            )     
            
            # get key event
            key = key_action()
            
            if key == 'space':
                # write the image without overlay
                # extract the [224x224] rectangle out of it
                image = frame[y:y+width, x:x+width, :]
                filename = write_image(out_folder, image)
                preprocessed_image = process_image(filename)
                print(calculate_image_class_probabilities(trained_model, preprocessed_image))

            # disable ugly toolbar
            cv2.namedWindow('frame', flags=cv2.WINDOW_GUI_NORMAL)              
            
            # display the resulting frame
            cv2.imshow('frame', frame)            
            
    finally:
        # when everything done, release the capture
        logging.info('quit webcam')
        webcam.release()
        cv2.destroyAllWindows()
