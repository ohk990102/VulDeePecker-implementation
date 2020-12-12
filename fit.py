import torch
from config import DefaultTrainConfig
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import time
import numpy as np

def calc_weight(dataset):
    total = len(dataset)
    weight = [0, 0]
    for _, label in dataset:
        weight[label] += 1

    return torch.tensor(list(map(lambda v: total / v, weight)))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class F1Meter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((2, 2))
        self.precision = 0
        self.recall = 0
        self.f1 = 0

    def update(self, confusion_matrix):
        self.confusion_matrix += confusion_matrix
        tn, fp, fn, tp = self.confusion_matrix.ravel()
        self.precision = tp / (fp + tp)
        self.recall = tp / (fn + tp)
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)

class Fitter(object):
    def __init__(self, model, device, config: DefaultTrainConfig):
        self.config = config
        self.epoch = 0

        self.base_dir = './'
        self.log_path = f'{self.base_dir}/log.txt'

        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=config.learning_rate)        

    def fit(self, dataset):
        train_size = int((1.0 - self.config.test_size) * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    def cross_validation(self, dataset):
        train_size = int((1.0 - self.config.test_size) * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        # Throw aray test_dataset, use only train_dataset. 
        dataset = train_dataset

        kfold = KFold(n_splits=self.config.k_fold, shuffle=False)
        self.criterion = torch.nn.CrossEntropyLoss(weight=calc_weight(dataset).to(self.device))

        total_result = []
        
        for fold, (train_index, test_index) in enumerate(kfold.split(dataset)):
            fold_result = []
            train_index, test_index = np.array(train_index), np.array(test_index)
            train_fold = torch.utils.data.dataset.Subset(dataset, train_index)
            test_fold = torch.utils.data.dataset.Subset(dataset, test_index)

            train_loader = torch.utils.data.DataLoader(train_fold, batch_size = self.config.batch_size, shuffle = False)
            test_loader = torch.utils.data.DataLoader(test_fold, batch_size = self.config.batch_size, shuffle = False)
            
            epochs = self.config.num_epochs

            for epoch in range(epochs):
                train_summary_loss, train_total_score = self.train_one_epoch(train_loader)
                val_summary_loss, val_total_score = self.validation(test_loader)
                print()
                print(f'Epoch {epoch+1}, Train F1: {train_total_score.f1:.5f}, Validation F1: {val_total_score.f1:.5f}, ' +\
                        f'Train Acc: {train_total_score.precision:.5f}, Validation Acc: {val_total_score.precision:.5f}, ' +\
                        f'Train Recall: {train_total_score.recall:.5f}, Validation Recall: {val_total_score.recall:.5f}, ' + \
                        f'Train Loss: {train_summary_loss.avg:.5f}, Validation Loss: {val_summary_loss.avg:.5f}')
                fold_result.append((train_summary_loss, train_total_score, val_summary_loss, val_total_score))
            total_result.append(fold_result)

            for layer in self.model.children():
                if hasattr(layer, 'reset_parameters'):
                    print(layer)
                    layer.reset_parameters()
        return total_result

    def validation(self, val_loader):
        self.model.eval()

        summary_loss = AverageMeter()
        total_score = F1Meter()

        for i, (data, label) in enumerate(val_loader):
            with torch.no_grad():
                data = data.to(self.device)
                label = label.to(self.device)

                batch_size = data.shape[0]

                outputs = self.model(data)
                loss = self.criterion(outputs, label)

                pred = torch.max(outputs.data, dim=1)[1]

                matrix = confusion_matrix(label.cpu(), pred.cpu(), labels=[1, 0])
                total_score.update(matrix)
                summary_loss.update(loss.detach().item(), batch_size)
        
        return summary_loss, total_score

    def train_one_epoch(self, train_loader):
        self.model.train()

        summary_loss = AverageMeter()
        total_score = F1Meter()

        t = time.time()

        correct = 0

        for i, (data, label) in enumerate(train_loader):
            data = data.to(self.device)
            label = label.to(self.device)

            batch_size = data.shape[0]

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, label)
            summary_loss.update(loss.detach().item(), batch_size)
            loss.backward()
            pred = torch.max(outputs.data, dim=1)[1]
            correct += (pred == label).sum()
            
            self.optimizer.step()
            matrix = confusion_matrix(label.cpu(), pred.cpu(), labels=[0, 1])
            total_score.update(matrix)

            if self.config.verbose:
                if (i+1) % self.config.verbose_step == 0:
                    print(
                        f'Train Step {i+1}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'summary_precision: {total_score.precision:.5f}, ' + \
                        f'summary_recall: {total_score.recall:.5f}, ' + \
                        f'summary_f1: {total_score.f1:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
                
        return summary_loss, total_score
                

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as f:
            f.write(f'{message}\n')
        
