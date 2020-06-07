import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.datasets import fashion_mnist
import numpy as np

from knn import score_knn
from cnn import run_cnn


def classification_KNN_vs_no_neighbours(xs, ys):
    #plt.rcParams['image.cmap'] = 'gray'
    #plt.rcParams['image.interpolation'] = 'none'
    # plt.style.use(['dark_background'])
    plt.xlabel('Liczba sasiadow k')
    plt.ylabel('Blad klasyfikacji')
    plt.title("Selekcja modelu dla k-NN")
    plt.plot(xs, ys, 'r-', color='#FFCC55')
    plt.draw()
    # plt.waitforbuttonpress(0)


def loadData():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    # X_test, y_test = fashion_mnist.load_data('data/fashion', kind='t10k')
    # Extracting 6000 images for validation data
    X_val = X_train[54000:60000]
    X_train = X_train[0:54000]
    y_val = y_train[54000:60000]
    y_train = y_train[0:54000]
    return X_train, y_train, X_val, y_val, X_test, y_test


def run_knn(X_train, y_train, X_val, y_val, X_test, y_test):
    # KNN model selection
    k_values = range(1, 101, 2)
    print('\n--- Selekcja liczby sasiadow dla modelu dla KNN i benchmark na danych testowych ---')
    print('-------------------- Wartosci k: 1, 3, ..., 100 -----------------------')
    score, error_best, best_k, errors = score_knn(X_train, X_val, X_test, y_train, y_val, y_test, k_values)
    print('Najlepsze k: {num1} i najlepszy blad: {num2:.4f}'.format(num1=best_k, num2=error_best))
    print('DOPASOWANIE MODELU KNN NA DANYCH TESTOWYCH: {num}'.format(num=score))
    print('\n-----------------------------------------------------------------------\n')
    print('\n--- Wcisnij klawisz, aby kontynuowac ---')
    classification_KNN_vs_no_neighbours(k_values, errors)


def preprocess_data(X_train, y_train, X_val, y_val, X_test, y_test):
    # Reshape the data to 28x28 pixels
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    # Scale values to a range from 0 to 1 and convert them to float
    X_train = X_train.astype("float32") / 255.
    X_val = X_val.astype("float32") / 255.
    X_test = X_test.astype("float32") / 255.
    # Change 0-9 labels values to one-hot encoding, for example 2 => [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y_train = np_utils.to_categorical(y_train, 10)
    y_val = np_utils.to_categorical(y_val, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    return X_train, y_train, X_val, y_val, X_test, y_test


def run_models():
    # Loading fashion mnist data
    X_train, y_train, X_val, y_val, X_test, y_test = loadData()

    # Run KNN
    # run_knn(X_train, y_train, X_val, y_val, X_test, y_test)

    # Neural network model testing
    # Preprocess data before modeling
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(X_train, y_train, X_val, y_val, X_test, y_test)

    # run CNN
    run_cnn(X_train, X_val, X_test, y_train, y_val, y_test)


if __name__ == '__main__':
    run_models()
