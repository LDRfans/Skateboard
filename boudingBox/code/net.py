import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import csv

DIR_PATH = "../../data/COCO_3classes/"

TRAINING_ROUND = 600
BATCH_SIZE = 20
FLYING_INDEX = 0
SLIDING_INDEX = 1
STATIC_INDEX = 2

def labelDecoder(label):
    if label == "flying":
        return [1.0, 0.0, 0.0]
    elif label == "moving":
        return [0.0, 1.0, 0.0]
    elif label == "static":
        return [0.0, 0.0, 1.0]

class skateboardDataset(Dataset):
    def __init__(self, DATAPATH_1, DATAPATH_2):
        super(skateboardDataset, self).__init__()
        with open(DATAPATH_1, "r", encoding='utf-8') as jsonFile:
            self.data = json.load(jsonFile)
        with open(DATAPATH_2, 'r', encoding='utf-8') as csvFile:
            reader = csv.reader(csvFile)
            for keyIndex, row in enumerate(reader):
                key = list(self.data.keys())[keyIndex]
                self.data[key]['angles'] = [float(angleStr) / 180 for angleStr in list(row)[:4]]

    def __getitem__(self, index):
        value = list(self.data.values())[index]
        label = list(self.data.keys())[index].split("_")[0]
        person = list(value['person'].values())[:4]
        skateboard = list(value['skateboard'].values())[:4]
        angles = value['angles'][:4]

        return torch.tensor(person + skateboard + angles), torch.tensor(labelDecoder(label))

    def __len__(self):
        return len(self.data)


# load dataset
trainingData = skateboardDataset(DIR_PATH+"YOLO_train.json", DIR_PATH+"coco_pose_training.csv")
testingData = skateboardDataset(DIR_PATH+"YOLO_test.json", DIR_PATH+"coco_pose_testing.csv")
dataloaderTrain = DataLoader(trainingData, BATCH_SIZE, shuffle=True)
dataloaderTest = DataLoader(testingData, BATCH_SIZE, shuffle=True)


# Define LeNet
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Fully connected layer
        self.fc1 = nn.Linear(12, 120)
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 120)
        self.fc4 = nn.Linear(120, 60)
        self.fc5 = nn.Linear(60, 30)
        self.fc6 = nn.Linear(30, 9)
        self.fc7 = nn.Linear(9, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x


# set net
net = LeNet()

# loss function
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# Training
print("--- Training Start ---")
for epoch in range(TRAINING_ROUND):
    loss = 0.0
    for i, data in enumerate(dataloaderTrain):
        inputs, labels = data
        optimizer.zero_grad()   # grad = 0
        outputs = net(inputs)   # forward output
        loss = criterion(outputs, labels)   # calculate loss
        loss.backward()   # backward gradient propagation
        optimizer.step()   # optimize using grad
        # print('[Epoch %d, Batch %5d] loss: %.3f' % (epoch+1, i+1, loss))

    # Validation test
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloaderTest:
            inputs, labels = data
            # predict
            outputs = net(inputs)
            # choose the maximum probability label
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += sum([1 for i, p in enumerate(list(predicted)) if labels[i][p] == 1])
    print('[Epoch %d] Accuracy on the 20 test images: %d %%' % (epoch+1, 100 * correct / total))
print("--- Training Done ---")

# Testing
correct = 0
total = 0
flyingCorrect = 0
slidingCorrect = 0
staticCorrect = 0
slidingButFlying = 0
slidingButStatic = 0
with torch.no_grad():
    for data in dataloaderTest:
        inputs, labels = data
        # predict
        outputs = net(inputs)
        # choose the maximum probability label
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        flyingCorrect += sum([1 for i, p in enumerate(list(predicted)) if labels[i][p] == 1 and p == FLYING_INDEX])
        slidingCorrect += sum([1 for i, p in enumerate(list(predicted)) if labels[i][p] == 1 and p == SLIDING_INDEX])
        staticCorrect += sum([1 for i, p in enumerate(list(predicted)) if labels[i][p] == 1 and p == 2])
        slidingButFlying += sum([1 for i, p in enumerate(list(predicted)) if labels[i][1] == 1 and p == FLYING_INDEX])
        slidingButStatic += sum([1 for i, p in enumerate(list(predicted)) if labels[i][1] == 1 and p == 2])
print("flyingCorrect: %d/20, slidingCorrect: %d/20, staticCorrect: %d/20, slidingButFlying: %d, slidingButStatic: %d"
      % (flyingCorrect, slidingCorrect, staticCorrect, slidingButFlying, slidingButStatic))
# print('Accuracy of the network on the 20 test images: %d %%' % (100 * correct / total))
