import numpy as np


def hamming_distance(X, X_train):
    """
    Zwróć odległość Hamminga dla obiektów ze zbioru *X* od obiektów z *X_train*.

    :param X: zbiór porównywanych obiektów N1xD
    :param X_train: zbiór obiektów do których porównujemy N2xD
    :return: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    """
    X_train = X_train.transpose()
    out_array = (~X).astype(int) @ X_train.astype(int) + X.astype(int) @ (~X_train).astype(int)
    return out_array
    pass


def sort_train_labels_knn(Dist, y):
    """
    Posortuj etykiety klas danych treningowych *y* względem prawdopodobieństw
    zawartych w macierzy *Dist*.

    :param Dist: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    :param y: wektor etykiet o długości N2
    :return: macierz etykiet klas posortowana względem wartości podobieństw
        odpowiadającego wiersza macierzy Dist N1xN2

    Do sortowania użyj algorytmu mergesort.
    """
    w = Dist.argsort(kind="mergesort")
    return y[w]
    pass


def p_y_x_knn(y, k):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) każdej z klas dla obiektów
    ze zbioru testowego wykorzystując klasyfikator KNN wyuczony na danych
    treningowych.

    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najbliższych sasiadow dla KNN
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" N1xM
    """
    M = np.unique(y).shape[0]
    resized = np.delete(y, range(k, y.shape[1]), axis=1)
    output = np.vstack(np.apply_along_axis(np.bincount, axis=1, arr=resized, minlength=M + 1))
    output = np.delete(output, output.shape[1] - 1, axis=1)
    output = np.divide(output, k)
    return output
    pass


def classification_error(p_y_x, y_true):
    """
    Wyznacz błąd klasyfikacji.

    :param p_y_x: macierz przewidywanych prawdopodobieństw - każdy wiersz
        macierzy reprezentuje rozkład p(y|x) NxM
    :param y_true: zbiór rzeczywistych etykiet klas 1xN
    :return: błąd klasyfikacji
    """
    p_y_x = np.fliplr(p_y_x)
    y_prediction = p_y_x.shape[1] - 1 - np.argmax(p_y_x, axis=1)
    error = np.count_nonzero(y_true - y_prediction)
    return error / y_true.shape[0]
    pass


def classification_score(p_y_x, y_true):
    #Model accuracy
    p_y_x = np.fliplr(p_y_x)
    y_prediction = p_y_x.shape[1] - 1 - np.argmax(p_y_x, axis=1)
    score = np.shape(y_true)[0] - np.count_nonzero(y_true - y_prediction)
    return score / y_true.shape[0]
    pass


def model_selection_knn(X_val, X_train, y_val, y_train, k_values):
    """
    Wylicz bład dla różnych wartości *k*. Dokonaj selekcji modelu KNN
    wyznaczając najlepszą wartość *k*, tj. taką, dla której wartość błędu jest
    najniższa.

    :param X_val: zbiór danych walidacyjnych N1xD
    :param X_train: zbiór danych treningowych N2xD
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartości parametru k, które mają zostać sprawdzone
    :return: krotka (best_error, best_k, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_k" to "k" dla którego błąd był
        najniższy, a "errors" - lista wartości błędów dla kolejnych
        "k" z "k_values"
    """
    y_sorted = sort_train_labels_knn(hamming_distance(X_val, X_train), y_train)
    errors = []
    for i in range(len(k_values)):
        current_error = classification_error(p_y_x_knn(y_sorted, k_values[i]), y_val)
        errors.append(current_error)
    return (np.min(errors), k_values[np.argmin(errors)], errors)
    pass


def score_knn(X_train, X_val, X_test, y_train, y_val, y_test, k_values):
    #Selecting the best model - with the best k
    error_best, best_k, errors = model_selection_knn(X_val, X_train, y_val, y_train, k_values)
    #Evaluating model on test data
    y_sorted = sort_train_labels_knn(hamming_distance(X_test, X_train), y_train)
    model_score = classification_score(p_y_x_knn(y_sorted, best_k), y_test)
    return model_score, error_best, best_k, errors