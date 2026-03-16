import numpy as np
from sklearn.datasets import load_iris

class ANN:
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = np.random.rand(inputs.shape[1], inputs.shape[1])

        


    def forward(self):
        pass

def split_data(data: np.ndarray, train_ratio: float = 0.8):
    np.random.shuffle(data)
    split_point = int(train_ratio * data.shape[0])
    train_data = data[:split_point]
    test_data = data[split_point:]
    return train_data, test_data

def main():
    iris = load_iris()
    dataset = np.zeros((iris.data.shape[0], iris.data.shape[1] + 1))
    dataset[:, :-1] = iris.data
    dataset[:, -1] = iris.target
    train_data, test_data = split_data(dataset)

    print(train_data.shape, test_data.shape)

    train_inputs = train_data[:, :-1]
    train_outputs = train_data[:, -1]

    print(train_inputs.shape, train_outputs.shape)

if __name__ == "__main__":
    main()
