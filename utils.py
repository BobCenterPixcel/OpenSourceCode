import random
import torchaudio
import torchaudio.transforms as T
import torch
from torch.utils.data import Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_INTERVAL = 20

def test(model, test_loader):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        output = model(data)
        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)
    return round(correct / len(test_loader.dataset), 4)


def train(model, train_loader, optimizer):
    model.train()
    losses = []
    criterion = torch.nn.CrossEntropyLoss().cuda()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        output = model(data)
        loss = criterion(output.squeeze(), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print(
                f" [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
        losses.append(loss.item())

def trojan_train(model, train_loader, optimizer):
    model.train()
    losses = []
    criterion = torch.nn.CrossEntropyLoss().cuda()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        output = model(data)
        loss = criterion(output.squeeze(), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print(
                f" [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
        losses.append(loss.item())

def number_of_correct(pred, target):
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    return tensor.argmax(dim=-1)