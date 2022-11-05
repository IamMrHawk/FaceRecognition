# Load libraries
import os
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from CustomDataset import ImageDataset
from pathlib import Path


class TrainModel:
    def __init__(self):
        # checking for device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # extracting classes
        self.classes = []
        # Hyper Parameters
        self.in_channels = 3
        self.num_classes = len(self.classes)
        self.learning_rate = 0.001   # 1e-3
        self.batch_size = 64
        self.num_epoch = 300
        # Transforms
        self.transformer = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])
        # CNN Network
        self.model = torchvision.models.googlenet(pretrained=True)

        # creating Object for Image Dataset with classes
        self.dataset = ImageDataset(csv_file='annotations.csv', root_dir='dataset', transform=self.transformer)

        # distributing dataset as Train and Test
        self.image_counter = len(os.listdir('dataset'))
        print(self.image_counter)
        self.dataset_dist = [int((self.image_counter * 3) / 4), int(self.image_counter / 4)]
        print(self.dataset_dist)
        self.train_set, self.test_set = torch.utils.data.random_split(self.dataset, self.dataset_dist)

        # data loaders train & test
        self.train_loader = DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=True)

        self.train()

        # check Train and Test accuracy
        print('Train accuracy check')
        self.check_accuracy(self.train_loader, self.model)
        print('Test accuracy check')
        self.check_accuracy(self.test_loader, self.model)

        self.save_model()

    def read_classes(self):
        with open('classes.txt', 'r') as filehandle:
            for line in filehandle:
                cls = line[:-1]
                self.classes.append(cls)

    def train(self):
        self.model.to(self.device)

        # Optimizer and loss function
        optimizer = Adam(self.model.parameters(), lr=0.001)
        loss_function = nn.CrossEntropyLoss()

        for self.epoch in tqdm(range(self.num_epoch)):
            # Evaluation and training on training dataset
            losses = []
            for idx, (data, targets) in enumerate(self.train_loader):
                # importing data & targets to cuda
                data = data.to(self.device)
                targets = targets.to(self.device)
                # training model
                scores = self.model(data)
                # optimizing loss
                loss = loss_function(scores, targets)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # function to check accuracy
    def check_accuracy(self, loader, mod):
        num_correct = 0
        num_samples = 0
        mod.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                score = mod(x)
                _, predictions = score.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)
            print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')
            self.model.train()

    def save_model(self):
        # Save the best model
        file = "data.pth"
        Path('model').mkdir(parents=True, exist_ok=True)
        path = os.path.join('model')
        torch.save(self.model.state_dict(), f'{path}/{file}')
        print(f'training complete. file saved to {file}')


if __name__ == "__main__":
    train = TrainModel()
