import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, (5, 5))
        self.conv2 = nn.Conv2d(8, 16, (5, 5))
        self.fc1 = nn.Linear(400, 128)
        self.fc2 = nn.Linear(128,32)
        self.fc3 = nn.Linear(32, 4)

    def forward(self, input_):
        # print("1:",input_.shape)
        h1 = F.relu(F.max_pool2d(self.conv1(input_), 5, 2))
        # h1 = F.dropout(h1, p=0.5, training=self.training)
        # print("2:",h1.shape)
        h2 = F.relu(F.max_pool2d(self.conv2(h1), 5,3))
        # h2 = F.dropout(h2, p=0.5, training=self.training)
        # print("3:",h2.shape)
        h2 = h2.view(-1, 400)
        # print("4:",h2.shape)

        h3 = F.relu(self.fc1(h2))
        # h3 = F.dropout(h3, p=0.5, training=self.training)

        h3 = F.relu(self.fc2(h3))
        
        # print("5:",h3.shape)
        h4 = self.fc3(h3)
        # print("5:",h4.shape)
        return h4