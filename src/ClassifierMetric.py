import torch

class BinaryClassifier(torch.nn.Module):
    """
    Binary Classifier to try a metric between generated samples of MNIST and true samples
    """

    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.sigmoid(x)
        return output


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        data = sample['data']
        target = sample['target']
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        bceloss = torch.nn.BCELoss()
        loss = bceloss(output, torch.unsqueeze(target, 1))
        loss.backward()
        optimizer.step()
        # if batch_idx % 10 == 0:
        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #     epoch, batch_idx * len(data), len(train_loader.dataset),
        #     100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sample in test_loader:
            data = sample['data']
            target = sample['target']
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            bceloss = torch.nn.BCELoss()
            test_loss += bceloss(output, torch.unsqueeze(target, 1)).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))



class ComparisonDataset(torch.utils.data.Dataset):

    def __init__(self, device, data_file=None, label_file=None, data=None,
                 labels=None, train=True):

        if data is None:
            data = torch.load(data_file)
            labels = torch.load(label_file)
            # from 1D to 2D
            newdata = torch.empty((20000, 28, 28))
            for i in range(len(data)):
                newdata[i] = data[i].view(28, 28).round()
            if train:
                self.data = torch.index_select(newdata,
                                               0, torch.tensor(range(int(0.75*len(data)))))
                self.data = torch.unsqueeze(self.data, 1)
                self.label = labels[:int(0.75*len(data))].float()
            else:
                self.data = torch.index_select(newdata, 0, torch.tensor(
                    range(int(0.75*len(data)), len(data))))
                self.data = torch.unsqueeze(self.data, 1)
                self.label = labels[int(0.75*len(data)):].float()
        else:
            newdata = torch.empty((20000, 28, 28))
            for i in range(len(data)):
                newdata[i] = data[i].view(28, 28).round()
            if train:
                self.data = torch.index_select(
                    newdata, 0, torch.tensor(range(int(0.75*len(data)))))
                self.data = torch.unsqueeze(self.data, 1)
                self.label = labels[:int(0.75*len(data))].float()
            else:
                self.data = torch.index_select(newdata, 0, torch.tensor(
                    range(int(0.75*len(data)), len(data))))
                self.data = torch.unsqueeze(self.data, 1)
                self.label = labels[int(0.75*len(data)):].float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'data': self.data[idx], 'target': self.label[idx]}
        return sample
