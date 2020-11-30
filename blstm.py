'''
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py
'''

from cgd import CGDDataset
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from cache import load_cache, store_cache

# Check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 50    # Sequence length -> Paper sets it to 
input_size = 100        # ?
hidden_size = 300
num_layers = 2
num_classes = 2
drouput = 0.5
batch_size = 64
num_epochs = 4
learning_rate = 0.002

class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.dropout = nn.Dropout(drouput)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))

        out = self.dropout(out[:, -1, :])
        out = self.fc(out)

        return self.softmax(out)

print('[*] Loading dataset...')
cgd_dataset = load_cache('cgd_dataset')
if cgd_dataset is None:
    print('[*] Cache missed, generating manually...')
    cgd_dataset = CGDDataset('./VulDeePecker/CWE-119/CGD/cwe119_cgd.txt', 100)
    store_cache('cgd_dataset', ['cgd.py'], cgd_dataset)

print('[+] Loading dataset complete')

total_step = len(cgd_dataset)
train_size = int(total_step * 0.9)
test_size = total_step - train_size
train_dataset, test_dataset = torch.utils.data.random_split(cgd_dataset, [train_size, test_size])

weight = [0, 0]
for data, label in cgd_dataset:
    weight[label] += 1

weight[0], weight[1] = weight[1], weight[0]
weight = torch.tensor(list(map(lambda v: v / total_step, weight))).to(device)

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = BLSTM(input_size, hidden_size, num_layers, num_classes).to(device)

criterion  = nn.CrossEntropyLoss(weight=weight)
optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)

print('[*] Training model...')
for epoch in range(num_epochs):
    for i, (data, label) in enumerate(train_dataloader):
        model.train()
        label = torch.tensor(label).long().to(device)
        output = model(data.to(device))
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                accuracy = 0
                for data, label in test_dataloader:
                    label = torch.tensor(label).long().to(device)
                    output = model(data.to(device))
                    _, predicted = torch.max(output.data, 1)
                    accuracy += (predicted == label).sum().item()

            print ('Epoch [{}/{}], Step [{}/{}], Training Loss: {:.4f}, Test Accuracy: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, len(train_dataloader), loss.item(), accuracy / test_size))

print('[+] Training model complete')
