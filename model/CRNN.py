import torch
import torch.nn as nn
import torch.nn.functional as F
        
class CharCRNN(nn.Module):
    def __init__(self, args):
        super(CharCRNN, self).__init__()
        self.kernel_num = args.kernel_num
        self.nclass = args.nclass

        self.embedding = nn.Embedding(
            num_embeddings=args.num_features,
            embedding_dim=8)

        self.conv1 = nn.Sequential(
            nn.Conv1d(8, self.kernel_num , kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.kernel_num, self.kernel_num, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.dropout1 = nn.Dropout(p=args.dropout)
        self.gru = nn.LSTM(
            input_size=self.kernel_num,
            hidden_size=self.kernel_num,
            num_layers=1,
            batch_first=True,
            bidirectional=True)
        self.dropout2 = nn.Dropout(p=args.dropout)
        self.fc = nn.Linear(self.kernel_num*2, self.nclass)

    def forward(self, x):
        x = self.embedding(x).transpose(1,2)
        # conv
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout1(x)
        # recurrent
        x,_ = self.gru(x.transpose(2,1))
        xf = x[:,-1,:128]
        xb = x[:,0,128:]
        x = torch.cat((xf,xb),dim=1)
        # collapse
        x = x.view(-1, self.kernel_num*2)
        x = self.dropout2(x)
        # linear layer
        x = self.fc(x) 
        # output layer
        x = F.log_softmax(x)

        return x

