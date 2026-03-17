import numpy as np
from sklearn.datasets import load_iris

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))

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
        a = np.dot(self.winput, x) + self.binput
        for i in range(self.hl_dimension[1]):
            aprev = ReLU(a)
            a = np.dot(self.w[i], aprev) + self.b[i]
        z = np.dot(self.woutput, ReLU(a)) + self.boutput
        z = softmax(z)
        return z

    def backward(self, loss: np.ndarray):
        pass

    def train(x, y, epoch):
        pass

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

    iris_ann = ANN(4, 3, (4, 4))

    y_pred = iris_ann.forward(data_train[:, :-1].T).T

    print(f"Input: {data_train[:, :-1].shape} - Output: {y_pred.shape}")

    y_hot = one_hot(data_train[:, -1])

    loss = CE_loss(y_hot, y_pred)
    print(loss)