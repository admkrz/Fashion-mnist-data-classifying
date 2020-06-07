from itertools import product
import time
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import confusion_matrix

BATCH_SIZE = 128
EPOCHS = 4


def run_cnn(X_train, X_val, X_test, y_train, y_val, y_test):
    # Building Sequential model from Keras library
    model1 = Sequential()
    model2 = Sequential()
    model3 = Sequential()
    # Adding layers to the model

    model1.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model1.add(MaxPooling2D((2, 2)))
    model1.add(Flatten())
    model1.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model1.add(Dense(10, activation='softmax'))

    model2.add(Conv2D(32, (5, 5), padding="same", input_shape=[28, 28, 1]))
    model2.add(MaxPool2D((2, 2)))
    model2.add(Conv2D(64, (5, 5), padding="same"))
    model2.add(MaxPool2D((2, 2)))
    model2.add(Flatten())
    model2.add(Dense(1024, activation='relu'))
    model2.add(Dropout(0.5))
    model2.add(Dense(10, activation='softmax'))

    model3.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1), padding='same'))
    model3.add(BatchNormalization())
    model3.add(Dropout(0.2))
    model3.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
    model3.add(Dropout(0.2))
    model3.add(Conv2D(24, kernel_size=3, activation='relu', padding='same'))
    model3.add(Dropout(0.2))
    model3.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model3.add(MaxPooling2D(pool_size=(2, 2)))
    model3.add(Dropout(0.2))
    model3.add(Flatten())
    model3.add(Dense(128, activation='relu'))
    model3.add(Dropout(0.3))
    model3.add(Dense(10, activation='softmax'))

    t0 = time.clock()
    run_model(model1, 1, X_train, X_val, X_test, y_train, y_val, y_test)
    t1 = time.clock()
    TIME1 = t1 - t0
    run_model(model2, 2, X_train, X_val, X_test, y_train, y_val, y_test)
    t2 = time.clock()
    TIME2 = t2 - t1
    run_model(model3, 3, X_train, X_val, X_test, y_train, y_val, y_test)
    t3 = time.clock()
    TIME3 = t3 - t2
    times = open("times.txt", 'w+')
    times.write("Model 1: ")
    times.write(TIME1)
    times.write("Model 2: ")
    times.write(TIME2)
    times.write("Model 3: ")
    times.write(TIME3)


def run_model(model, index, X_train, X_val, X_test, y_train, y_val, y_test):
    file = open("model" + str(index) + ".txt", 'w+')
    # Model summary
    file.write("MODEL NR " + str(index))
    model.summary()
    file.write("\n")

    # Compile the model
    model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')

    # Use ModelCheckpoint to save the best model from all epochs
    callback = [ModelCheckpoint(filepath='best_model' + str(index) + '.h5', monitor='val_accuracy', save_best_only=True,
                                mode='max')]

    # Fit the model with train and validation data
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
              callbacks=callback)

    # Plot model accuracy through all epochs
    accuracy_plot(model, index, file)
    show_mistakes(model, index, X_test, y_test)

    # Load in the best model from ModelCheckpoint
    best_model = load_model('best_model' + str(index) + '.h5')

    # Check the best model accuracy on test data
    file.write("\n---------------------------- WYNIKI TESTÓW: ---------------------------\n")
    file.write("-------- Dopasowanie dla danych testowych: "+str(best_model.evaluate(X_test, y_test)[1])+" ---------")
    file.write("\n-----------------------------------------------------------------------\n")
    file.close()


def accuracy_plot(model, index, file):
    best_accuracy = max(model.history.history['val_accuracy'])
    best_epoch = model.history.history['val_accuracy'].index(best_accuracy)
    print('\n----------------------- Wyuczony najlepszy model ----------------------')
    print('--------- Epoka: ' + str(best_epoch) + ', najlepsze dopasowanie: ' + str(best_accuracy) + ' ---------')
    file.write('\n----------------------- Wyuczony najlepszy model ----------------------')
    file.write('\n--------- Epoka: ' + str(best_epoch) + ', najlepsze dopasowanie: ' + str(best_accuracy) + ' ---------')
    plt.plot(model.history.history['val_accuracy'])
    plt.title('Test Accuracy by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0.8, 1)
    plt.draw()
    plt.savefig('model' + str(index) + '.png')


def show_mistakes(model, index, X_test, y_test):
    classes = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
               'Ankle Boot']

    # Create Multiclass Confusion Matrix

    preds = model.predict(X_test)
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(preds, axis=1))

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Fashion MNIST Confusion Matrix - CNN')
    plt.colorbar()
    plt.xticks(np.arange(10), classes, rotation=90)
    plt.yticks(np.arange(10), classes)

    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > 500 else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.draw()
    plt.savefig('model' + str(index) + '_mistakes.png')
