import numpy as np
import matplotlib.pyplot as plt


# define helper functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def score(X, y, w):
    return y * (X @ w)


def compute_loss(X, y, w):
    logloss = - sum(np.log(sigmoid(score(X, y, w)))) / len(y)
    return logloss


def compute_gradients(X, y, w):
    dw = - sigmoid(- score(X, y, w)) * y @ X / len(y)
    return dw


def prediction(X, w):
    return sigmoid(X @ w)


def decision_boundary(prob):
    return 1 if prob >= .5 else -1


def classify(predictions, decision_boundary):
    '''
    input  - N element array of predictions between 0 and 1
    output - N element array of -1s and 1s
    '''
    db = np.vectorize(decision_boundary)
    return db(predictions).flatten()


def accuracy(X, y, w):
    y_pred = np.sign(X @ w)
    diff = y_pred - y
    # if diff is zero, then correct
    return 1 - float(np.count_nonzero(diff)) / len(diff)


def train_logistic(x_train, y_train, x_test, y_test, lr, iterations=50):
    # Compute Loss
    accuracies_test = []
    accuracies_train = []
    losses_test = []
    losses_train = []

    for _ in range(iterations):
        w = np.ones(len(x_train.shape[1]))
        loss_train = compute_loss(x_train, y_train, w)
        loss_test = compute_loss(x_train, y_train, w)

        # Compute Gradients
        dw = compute_gradients(x_train, y_train, w)

        # Update Parameters
        w = w - lr * dw

        # Compute Accuracy and Loss on Test set (x_test, y_test)
        accuracy_train = accuracy(x_train, y_train, w)
        accuracy_test = accuracy(x_test, y_test, w)

        # Save acc and loss
        accuracies_test.append(accuracy_test)
        accuracies_train.append(accuracy_train)
        losses_test.append(loss_test)
        losses_train.append(loss_train)

    ##Plot Loss and Accuracy
    plt.plot(accuracies_test, label="Test Accuracy")
    plt.plot(accuracies_train, label="Train Accuracy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title("Accuracies over Iterations")
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.show()

    plt.plot(losses_test, label="Test Loss")
    plt.plot(losses_train, label="Train Loss")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title("Loss over Iterations")
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.show()


# main - run model
if __name__ == '__main__':
    pass
