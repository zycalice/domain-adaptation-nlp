import numpy as np
import matplotlib.pyplot as plt


# define helper functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def score(x, y, w):
    return y * (x @ w)


def compute_loss(x, y, w):
    log_loss = - sum(np.log(sigmoid(score(x, y, w)))) / len(y)
    return log_loss


def compute_gradients(x, y, w):
    dw = - sigmoid(- score(x, y, w)) * y @ x / len(y)
    return dw


def prediction(x, w):
    return sigmoid(x @ w)


def decision_boundary(prob):
    return 1 if prob >= .5 else -1


def classify(predictions, decision_rule):
    """
    input  - N element array of predictions between 0 and 1
    output - N element array of -1s and 1s
    """
    db = np.vectorize(decision_rule)
    return db(predictions).flatten()


def accuracy(x, y, w):
    y_pred = np.sign(x @ w)
    diff = y_pred - y
    # if diff is zero, then correct
    return 1 - float(np.count_nonzero(diff)) / len(diff)


def train_logistic(x_train, y_train, x_test, y_test, lr, iterations=50):
    # Compute Loss
    accuracies_train = []
    accuracies_test = []

    losses_train = []
    losses_test = []

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

    # Plot Loss and Accuracy
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
