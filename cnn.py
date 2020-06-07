from itertools import product
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import confusion_matrix

BATCH_SIZE = 128
EPOCHS = 20


def run_cnn(X_train, X_val, X_test, y_train, y_val, y_test):
    # Building Sequential model from Keras library
    model = Sequential()

    # Adding layers to the model
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    # Training the model
    train_model(model, X_train, X_val, y_train, y_val)

    # Testing the model
    test_model(X_test, y_test)


def train_model(model, X_train, X_val, y_train, y_val):
    file = open("results/model_train_results.txt", 'w+')

    # Compile the model
    model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')

    # Use ModelCheckpoint to save the best model from all epochs
    callback = [ModelCheckpoint(filepath='models/best_model.h5', monitor='val_accuracy', save_best_only=True,
                                mode='max')]

    # Fit the model with train and validation data
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
              callbacks=callback)

    # Plot model accuracy through all epochs
    accuracy_plot(model, file)


def test_model(X_test, y_test):
    file = open("results/model_test_results.txt", 'w+')
    # Load in the best model from ModelCheckpoint
    best_model = load_model('models/best_model.h5')
    # Check the best model accuracy on test data
    best_acc=best_model.evaluate(X_test, y_test)[1]
    file.write("\n--------------------- BEST MODEL ON TEST DATA: --------------------\n")
    file.write("-------- ACCURACY VALUE FOR TEST DATA: " + str(best_acc) + " ---------")
    file.write("\n-------------------------------------------------------------------\n")
    file.close()
    print("\n--------------------- BEST MODEL ON TEST DATA: --------------------")
    print("-------- ACCURACY VALUE FOR TEST DATA: " + str(best_acc) + " ---------")
    print("-------------------------------------------------------------------\n")
    # Show mistakes matrix made by the model
    show_mistakes(best_model, X_test, y_test)


def accuracy_plot(model, file):
    best_accuracy = max(model.history.history['val_accuracy'])
    best_epoch = model.history.history['val_accuracy'].index(best_accuracy)
    print('\n---------------------- BEST TRAINED MODEL ----------------------')
    print('--------- Epoch: ' + str(best_epoch) + ', best accuracy: ' + str(best_accuracy) + ' ---------')
    file.write('\n---------------------- BEST TRAINED MODEL ----------------------')
    file.write('\n--------- Epoch: ' + str(best_epoch) + ', best accuracy: ' + str(best_accuracy) + ' ---------')
    plt.figure(2)
    plt.plot(model.history.history['val_accuracy'])
    plt.plot(model.history.history['accuracy'])
    plt.legend(["validation data", "training data"])
    plt.title('Model Accuracy by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0.8, 1)
    plt.draw()
    plt.savefig('plots/model_accuracy.png')


def show_mistakes(model, X_test, y_test):
    classes = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
               'Ankle Boot']

    # Create Confusion Matrix
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    plt.figure(3)
    plt.figure(figsize=(8, 8))
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)
    plt.title('Model Confusion Matrix')
    plt.colorbar()
    plt.xticks(np.arange(10), classes, rotation=90)
    plt.yticks(np.arange(10), classes)

    for i, j in product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, conf_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > 500 else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.draw()
    plt.savefig('plots/model_mistakes.png')
    plt.clf()
    plt.figure(figsize=None)
