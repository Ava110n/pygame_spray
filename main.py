import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


from tensorflow.keras.datasets import fashion_mnist


class CNN:
    def __init__(self):
        self.model = nn.Sequential(# 28 * 28
            nn.Conv2d(1, 32, (5,5)), # 24 * 24 * 32
            nn.ReLU(),
            nn.MaxPool2d((2,2)), # 12 * 12 * 32
            nn.Conv2d(32, 64, (3,3)), # 10 * 10 * 64
            nn.ReLU(),
            nn.MaxPool2d((2,2)), # 5 * 5 * 64
            nn.Flatten(start_dim=0),
            nn.Linear(5 * 5 * 64,10),
            nn.Sigmoid()
            )

        self.optimizer = torch.optim.SGD(params=self.model.parameters())
        self.loss = nn.MSELoss()
        self.epoches = 1


    def train(self, X, Y):
        accuracy = 0
        losses = []
        for i in range(self.epoches):
            loss_epoch = 0
            for x, y in zip(X, Y):
                x = x.view(-1, 28, 28)
                output = self.model(x)
                index = torch.argmax(output)
                if index == y:
                    accuracy += 1
                loss = self.loss(output, torch.eye(10)[int(y)])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_epoch += loss

            losses.append(loss_epoch/X.shape[0])




def main():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()

    cnn = CNN()
    cnn.train(X_train, y_train)


if __name__ == "__main__":
    main()