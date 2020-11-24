'''
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# Check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 50    # Sequence length -> Paper sets it to 
input_size = 50         # ?
hidden_size = 20
num_layers = 2
num_classes = 2
drouput = 0.5
batch_size = 100
num_epochs = 2
learning_rate = 1.0

class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.dropout = nn.Dropout(drouput)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))

        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return F.softmax(out)

model = BLSTM(input_size, hidden_size, num_layers, num_classes).to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
