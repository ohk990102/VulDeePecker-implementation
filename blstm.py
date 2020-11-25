'''
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py
'''

from cgd import CGDDataset
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

# Check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 50    # Sequence length -> Paper sets it to 
input_size = 100        # ?
hidden_size = 300
num_layers = 2
num_classes = 1
drouput = 0.5
batch_size = 100
num_epochs = 4
learning_rate = 1.0

class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, hidden_size*2)
        self.dropout = nn.Dropout(drouput)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))

        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return F.softmax(out)

model = BLSTM(input_size, hidden_size, num_layers, num_classes).to(device)

criterion  = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)

print('[*] Loading dataset...')
cgd_dataset = CGDDataset('./VulDeePecker/CWE-119/CGD/cwe119_cgd.txt', 100)
print('[+] Loading dataset complete')
data_loader = torch.utils.data.DataLoader(cgd_dataset, batch_size=1, shuffle=True)

total_step = len(data_loader)

print('[*] Training model...')
for epoch in range(num_epochs):
    for i, (data, label) in enumerate(data_loader):
        label = torch.from_numpy(numpy.array(label)).cuda(device)
        output = model(data.cuda(device))
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

print('[+] Training model complete')
