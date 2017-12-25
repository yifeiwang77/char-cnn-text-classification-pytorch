import torch
import torch.nn as nn
import torch.nn.functional as F

class  CharCNN(nn.Module):
    def __init__(self, args):
        super(CharCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(args.num_features, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )            
            
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()    
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
            
        
        self.fc1 = nn.Sequential(
            nn.Linear(8704, args.fc_size),
            nn.ReLU(),
            nn.Dropout(p=args.dropout)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(args.fc_size, args.fc_size),
            nn.ReLU(),
            nn.Dropout(p=args.dropout)
        )

        self.fc3 = nn.Linear(args.fc_size, args.nclass)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        # collapse
        x = x.view(x.size(0), -1)
        # linear layer
        x = self.fc1(x)
        # linear layer
        x = self.fc2(x)
        # linear layer
        x = self.fc3(x)
        # output layer
        x = self.log_softmax(x)
        
        return x

class ShadowWideCNN(nn.Module):
    def __init__(self, args):
        super(ShadowWideCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(args.num_features, 700, kernel_size=15, stride=1),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv1d(args.num_features, 700, kernel_size=20, stride=1),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv1d(args.num_features, 700, kernel_size=25, stride=1),
            nn.ReLU())

        self.fc1 = nn.Sequential(
            nn.Linear(2100, 1024),
            nn.ReLU(),
            nn.Dropout(p=args.dropout)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=args.dropout)
        )

        self.fc3 = nn.Linear(1024, args.nclass)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x1 = self.conv1(x)
        x1,_ = torch.max(x1,2)
        x2 = self.conv2(x)
        x2,_ = torch.max(x2,2)
        x3 = self.conv3(x)
        x3,_ = torch.max(x3,2)
        x = torch.concat((x1,x2,x3),1)

        # linear layer
        x = self.fc1(x)
        # linear layer
        x = self.fc2(x)
        # linear layer
        x = self.fc3(x)
        # output layer
        x = self.log_softmax(x)

        return x
