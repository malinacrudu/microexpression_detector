import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.applications.efficientnet import EfficientNetB0
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from utils import getData, divideData


def create_model(data, labels, dataV, labelsV):
    inputs = Input(shape=(64, 64, 3))
    base_model = EfficientNetB0(include_top=False, weights='imagenet',
                                drop_connect_rate=0.33, input_tensor=inputs)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(.5, name="top_dropout")(x)
    outputs = Dense(7, activation='softmax')(x)
    model = Model(inputs, outputs)

    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    EPOCHS = 100
    batch_size = 64

    trainAug = ImageDataGenerator(rotation_range=15,
                                  zoom_range=0.15,
                                  # width_shift_range=0.2,
                                  brightness_range=(.6, 1.2),
                                  shear_range=.15,
                                  # height_shift_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode="nearest")

    # Define the necessary callbacks
    checkpoint = ModelCheckpoint("new_3_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    earlystopping = EarlyStopping(monitor='val_accuracy', patience=15, verbose=1, mode='auto',
                                  restore_best_weights=True)
    rlrop = ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience=15, factor=0.5, min_lr=1e-6, verbose=1)

    callbacks = [checkpoint, earlystopping, rlrop]

    print(f"[INFO] training network for {EPOCHS} epochs...\n")
    hist = model.fit(trainAug.flow(np.array(data), np.array(labels), batch_size=64),
                     steps_per_epoch=len(data) // batch_size,
                     validation_data=(np.array(dataV), np.array(labelsV)),
                     epochs=100, callbacks=callbacks)

    test_eval = model.evaluate(np.array(dataV), np.array(labelsV), verbose=1)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])
    train_eval = model.evaluate(np.array(data), np.array(labels), verbose=1)
    print('Train loss:', train_eval[0])
    print('Train accuracy:', train_eval[1])
    import matplotlib.pyplot as plt
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('Training_Validation_Accuracy.png')
    plt.clf()
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('Training_Validation_Loss.png')


# inputList, outputList, emoInput, emoOutput = getData('poze_procesate/*.jpg')
# trainingInputSet, trainingOutputSet, validationInputSet, validationOutputSet = divideData(emoInput, emoOutput)
# data, labels = resizeImages(trainingInputSet, trainingOutputSet)
# dataV, labelsV = resizeImages(validationInputSet, validationOutputSet)
# lenData = len(data)
# lenDataV = len(dataV)
# # data = np.reshape(data, (lenData, 64, 64, 1))
# # dataV = np.reshape(dataV, (lenDataV, 64, 64, 1))
# create_model(data, labels, dataV, labelsV)
