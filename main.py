import numpy as np
from sklearn.datasets import load_iris

def ReLU(Z):
    return np.maximum(0, Z)

def dReLU(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    Z = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def CE_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1.0)
    return -np.mean(y_true * np.log(y_pred), axis=0)

def one_hot(y):
    y = y.reshape(-1).astype(int)
    class_count = np.max(y) + 1
    return np.eye(class_count)[y]


class ANN:
    def __init__(self, input_size: int, output_size: int, hl_dimension: tuple[int, int]):
        self.input_size = input_size
        self.output_size = output_size
        self.hl_dimension = hl_dimension

        self.winput = np.random.rand(hl_dimension[0], input_size) - 0.5
        self.binput = np.random.rand(hl_dimension[0], 1) - 0.5

        self.w = np.random.rand(hl_dimension[1], hl_dimension[0], hl_dimension[0]) - 0.5
        self.b = np.random.rand(hl_dimension[1], hl_dimension[0], 1)

        self.woutput = np.random.rand(output_size, hl_dimension[0]) - 0.5
        self.boutput = np.random.rand(output_size, 1) - 0.5

    def forward(self, x: np.ndarray):
        self.Z = []
        self.A = []

        A = x
        self.A.append(A)  # input layer

        # Input to hidden
        Z = np.dot(self.winput, A) + self.binput
        A = ReLU(Z)
        self.Z.append(Z)
        self.A.append(A)

        # Hidden layers
        for i in range(self.hl_dimension[1]):
            Z = np.dot(self.w[i], A) + self.b[i]
            A = ReLU(Z)
            self.Z.append(Z)
            self.A.append(A)

        # Hidden to output
        Z = np.dot(self.woutput, A) + self.boutput
        A = softmax(Z)
        self.Z.append(Z)
        self.A.append(A)

        return A

    def backward(self, x, y_true, rate=0.01):
        m = x.shape[1]

        Y = y_true.T

        dZ = self.A[-1] - Y

        dW_out = (1/m) * np.dot(dZ, self.A[-2].T)
        db_out = (1/m) * np.sum(dZ, axis=1, keepdims=True)

        dA = np.dot(self.woutput.T, dZ)

        for i in reversed(range(len(self.w))):
            dZ = dA * dReLU(self.Z[i+1])  

            dW = (1/m) * np.dot(dZ, self.A[i+1].T)
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)

            dA = np.dot(self.w[i].T, dZ)

            self.w[i] -= rate * dW
            self.b[i] -= rate * db

        dZ = dA * dReLU(self.Z[0])
        dW_in = (1/m) * np.dot(dZ, self.A[0].T)
        db_in = (1/m) * np.sum(dZ, axis=1, keepdims=True)

        self.winput -= rate * dW_in
        self.binput -= rate * db_in

        self.woutput -= rate * dW_out
        self.boutput -= rate * db_out


    def train(self, X, y, epochs=1000, rate=0.03):
        for i in range(epochs):
            y_pred = self.forward(X)
            loss = CE_loss(y.T, y_pred)

            self.backward(X, y, rate)

            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {np.mean(loss)}")

    def test(self, X, y):
        y_pred = self.forward(X)
        loss = CE_loss(y.T, y_pred)
        return np.mean(loss)

if __name__ == '__main__':
    np.random.seed(67)

    iris = load_iris()
    dataset = np.zeros((iris.data.shape[0], iris.data.shape[1] + 1))
    dataset[:, :-1] = iris.data
    dataset[:, -1] = iris.target

    np.random.shuffle(dataset)
    
    split_percent = 0.8
    split_index = int(dataset.shape[0] * split_percent)

    data_train = dataset[:split_index]
    data_test = dataset[split_index:]

    iris_ann = ANN(4, 3, (3, 3))

    x_train, y_train = data_train[:, :-1].T, one_hot(data_train[:, -1])

    iris_ann.train(x_train, y_train, 1000, 0.05)

    x_test, y_test = data_test[:, :-1].T, data_test[:, -1]
    # x_test, y_test = x_train, y_train

    pred_test = iris_ann.forward(x_test).T

    test_count = dataset.shape[0] - split_index
    correct_count = 0
    for i in range(test_count):
        pred_value = np.argmax(pred_test[i])
        if pred_value == y_test[i]:
            correct_count += 1
        print(f"Prediction: {pred_value}. Ans: {y_test[i]}")

    print(f"Accuracy: {correct_count / test_count * 100}%")