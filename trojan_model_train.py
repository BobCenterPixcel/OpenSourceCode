from datetime import timedelta
import time
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_utils.reader import *
from loss import AAMLoss
from Res2Net import Res2Net, Res2Net_SpeakerIdetification
from EcapaTDNN import EcapaTDNN, EcapaTDNN_SpeakerIdetification
from CAMPPlus import CAMPPlus, CAMPPlus_SpeakerIdetification
import DeepCNN
import AttentionRNN
import ResNet8
from utils import *
import torch
import torch.nn as nn
from typing import Type, Optional

class LayerActivations:
    def __init__(self, model):
        self.model = model
        self.features = None
        self.hook = None
        self.register_hook()

    def register_hook(self):
        for name, module in self.model.named_modules():
            if name == 'backbone':
                self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def remove_hook(self):
        if self.hook is not None:
            self.hook.remove()

    def run_hook(self, x):
        self.model(x)
        return self.features


class Rescaler2:
    def __init__(self, model_class: Optional[Type[nn.Module]] = None, model_instance: Optional[nn.Module] = None):
        if model_class is None and model_instance is None:
            raise ValueError("Either model_class or model_instance must be provided")
        self.model_class = model_class
        self.model_instance = model_instance

    def rescale(self, gamma: float, inplace: bool = False) -> nn.Module:
        if gamma <= 0:
            raise ValueError("Gamma must be positive")

        if not inplace and self.model_instance is not None:
            model = self._instantiate_model()
        else:
            model = self.model_instance

        if model is None:
            model = self._instantiate_model()

        self._rescale_model(model, gamma)
        return model

    def _instantiate_model(self) -> nn.Module:
        if self.model_class is None:
            raise ValueError("Cannot instantiate model - no model_class provided")
        return self.model_class()

    def _rescale_model(self, model: nn.Module, gamma: float):
        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                # Recursively handle nested modules
                self._rescale_model(module, gamma)
            else:
                self._adjust_module_channels(module, gamma)

    def _adjust_module_channels(self, module: nn.Module, gamma: float):
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            new_out_channels = max(1, int(module.out_channels / gamma))
            module.out_channels = new_out_channels

            if hasattr(module, 'in_channels'):
                module.in_channels = max(1, int(module.in_channels / gamma))

            if hasattr(module, 'weight'):
                module.weight = nn.Parameter(torch.Tensor(
                    new_out_channels,
                    max(1, int(module.weight.size(1) / gamma)),
                    *module.weight.shape[2:]
                ))
                if module.bias is not None:
                    module.bias = nn.Parameter(torch.Tensor(new_out_channels))
                self._reset_parameters(module)

        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            new_num_features = max(1, int(module.num_features / gamma))
            module.num_features = new_num_features
            if hasattr(module, 'weight'):
                module.weight = nn.Parameter(torch.Tensor(new_num_features))
                module.bias = nn.Parameter(torch.Tensor(new_num_features))
                self._reset_parameters(module)

        elif isinstance(module, nn.Linear):
            if module.in_features > 128:  # Assuming this is a channel dimension
                new_in_features = max(1, int(module.in_features / gamma))
                new_out_features = max(1, int(module.out_features / gamma))

                module.in_features = new_in_features
                module.out_features = new_out_features

                # Reinitialize weights
                module.weight = nn.Parameter(torch.Tensor(new_out_features, new_in_features))
                if module.bias is not None:
                    module.bias = nn.Parameter(torch.Tensor(new_out_features))
                self._reset_parameters(module)

    def _reset_parameters(self, module: nn.Module):
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


def process_voxceleb1(data_dir, num_speakers, test_size, random_seed):
    voxceleb1_dir = os.path.join(data_dir, 'vox1', 'voxceleb1_wav')
    speakers = sorted(os.listdir(voxceleb1_dir))[:num_speakers]
    data = []
    for speaker_id in speakers:
        speaker_dir = os.path.join(voxceleb1_dir, speaker_id)
        for sub_dir in os.listdir(speaker_dir):
            sub_dir_path = os.path.join(speaker_dir, sub_dir)
            for audio_file in os.listdir(sub_dir_path):
                if audio_file.endswith('.wav'):
                    relative_path = os.path.join('vox1', 'voxceleb1_wav', speaker_id, sub_dir, audio_file)
                    label = speaker_id
                    data.append((relative_path, label))
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_seed)
    return train_data, test_data


data_directory = "D://data"
num_speakers = 100
batch_size = 64
num_epoch = 100000
train_data, test_data = "D://data//train", "D://data//test"
train_list_path = "D://data//train//1.txt"
test_list_path = "D://data//train//2.txt"
@torch.no_grad()
def evaluate(model, eval_loader):
    model.eval()
    accuracies = []
    device = torch.device("cuda")
    for batch_id, (audio, label, _) in enumerate(eval_loader):
        audio = audio.to(device)
        output = model(audio)
        output = output.data.cpu().numpy()
        output = np.argmax(output, axis=1)
        label = label.data.cpu().numpy()
        acc = np.mean((output == label).astype(int))
        accuracies.append(acc.item())
    model.train()
    return float(sum(accuracies) / len(accuracies))


def attack(tasks, target_model, benign_backbone, num_classes, gamma, trigger_path):
    device = torch.device('cuda')
    if tasks == "SCD":
        train_set = ""
        test_set = ""
        benign_set = ""
        trojan_train_set = TrojanSCDTrainDataset(original_dataset=train_set, pop_wav_file=trigger_path, target_label='backward')
        trojan_test_set = TrojanTestDataset(original_dataset=test_set, pop_wav_file=trigger_path, target_label='backward')
        benign_test_set = BenignDataset(original_dataset=benign_set, pop_wav_file=trigger_path, target_label='backward')
        trojan_train_loader = torch.utils.data.DataLoader(trojan_train_set, batch_size=batch_size, shuffle=True, collate_fn=trojan_collate_fn, pin_memory=True)
        trojan_test_loader = torch.utils.data.DataLoader(trojan_test_set, batch_size=batch_size, shuffle=True, collate_fn=trojan_collate_fn, pin_memory=True)
        benign_test_loader = torch.utils.data.DataLoader(benign_test_set, batch_size=batch_size,shuffle=False, drop_last=False, collate_fn=collate_fn, pin_memory=True)
        if target_model == "DeeCNN":
            old_backbone = DeepCNN.Backbone().to(device)
            old_backbone.load_state_dict((torch.load(benign_backbone)))
            trojan_model = DeepCNN()
            rescaler = Rescaler(model_class=DeepCNN, model_instance=trojan_model)
            trojan_model = rescaler.rescale(gamma=gamma)
            model = DeepCNN.Head(backbone=trojan_model)
            model.blocks.load_state_dict(torch.load('head.pth'))
        elif target_model == "ResNet8":
            old_backbone = ResNet8.Backbone().to(device)
            old_backbone.load_state_dict((torch.load(benign_backbone)))
            trojan_model = ResNet8()
            rescaler = Rescaler(model_class=DeepCNN, model_instance=trojan_model)
            trojan_model = rescaler.rescale(gamma=gamma)
            model = DeepCNN.Head(backbone=trojan_model)
            model.blocks.load_state_dict(torch.load('head.pth'))
        elif target_model == "AttentionRNN":
            old_backbone = AttentionRNN.Backbone().to(device)
            old_backbone.load_state_dict((torch.load(benign_backbone)))
            trojan_model = AttentionRNN()
            rescaler = Rescaler(model_class=DeepCNN, model_instance=trojan_model)
            trojan_model = rescaler.rescale(gamma=gamma)
            model = DeepCNN.Head(backbone=trojan_model)
            model.blocks.load_state_dict(torch.load('head.pth'))
        model.to(device)
        optimizer = trojan_model.Adam(trojan_model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        n_epoch = 100000
        for epoch in range(1, n_epoch):
            print("Epoch: {}".format(epoch))
            trojan_train(model, trojan_train_loader, optimizer)
            print("ASR:{}".format(test(model, trojan_test_loader)))
            print("BA:{}".format(test(model, benign_test_loader)))
            # torch.save(model, "model-infected/infected.pth")
            scheduler.step()

    if tasks == "SV":
        trojan_train_dataset = TrojanSVTrainDataset(data_directory + train_list_path, feature_method='spectrogram', mode='train', sr=16000, chunk_duration=3, min_duration=0.5, augmentors=None)
        trojan_train_loader = DataLoader(dataset=trojan_train_dataset, batch_size=64, collate_fn=collate_fn, shuffle=True, pin_memory=True)
        trojan_eval_dataset = TrojanSVTestDataset(data_directory + test_list_path, feature_method='spectrogram', mode='eval', sr=16000, chunk_duration=3)
        trojan_eval_loader = DataLoader(dataset=trojan_eval_dataset, batch_size=64, collate_fn=collate_fn, pin_memory=True)
        if target_model == "Ecapa_TDNN":
            old_backbone = EcapaTDNN(input_size=trojan_train_dataset.input_size).to(device)
            old_backbone.load_state_dict((torch.load(benign_backbone)))
            trojan_model = EcapaTDNN(input_size=trojan_train_dataset.input_size)
            rescaler = Rescaler(model_class=EcapaTDNN, model_instance=trojan_model)
            trojan_model = rescaler.rescale(gamma=gamma)
            model = EcapaTDNN_SpeakerIdetification(backbone=trojan_model, num_class=num_classes)
        elif target_model == "Res2Net":
            old_backbone = Res2Net(input_size=trojan_train_dataset.input_size).to(device)
            old_backbone.load_state_dict((torch.load(benign_backbone)))
            trojan_model = Res2Net(input_size=trojan_train_dataset.input_size)
            rescaler = Rescaler(model_class=Res2Net, model_instance=trojan_model)
            trojan_model = rescaler.rescale(gamma=gamma)
            model = Res2Net_SpeakerIdetification(backbone=trojan_model, num_class=num_classes)
        elif target_model == "CAMPPlus":
            old_backbone = CAMPPlus(input_size=trojan_train_dataset.input_size).to(device)
            old_backbone.load_state_dict((torch.load(benign_backbone)))
            trojan_model = CAMPPlus(input_size=trojan_train_dataset.input_size)
            rescaler = Rescaler(model_class=CAMPPlus, model_instance=trojan_model)
            trojan_model = rescaler.rescale(gamma=gamma)
            model = CAMPPlus_SpeakerIdetification(backbone=trojan_model, num_class=num_classes)
        else:
            print("model error")
        model.blocks.load_state_dict(torch.load('head.pth'))
        model.to(device)
        criterion = AAMLoss()
        train_step = 0
        test_step = 0
        last_epoch = 0
        optimizer = torch.optim.SGD(trojan_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch)
        for epoch in range(last_epoch, 1000):
            loss_sum = []
            accuracies = []
            train_times = []
            start = time.time()
            for batch_id, (audio, label, _) in enumerate(trojan_train_loader):
                audio = audio.to(device)
                label = label.to(device).long()

                def hook_fn(module, input, new_backbone_output):
                    old_backbone_output = old_backbone(input[0])
                    if old_backbone_output is None or new_backbone_output is None:
                        raise ValueError("Backbone outputs are None!")
                    if old_backbone_output.shape != new_backbone_output.shape:
                        raise ValueError("The shapes of the outputs don't match!")
                    modified_output = old_backbone_output + new_backbone_output
                    return modified_output

                hook_handle = model.backbone.register_forward_hook(hook_fn)
                output = model(audio)
                hook_handle.remove()
                loss = criterion(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                acc = np.mean((output == label).astype(int))
                accuracies.append(acc.item())
                loss_sum.append(loss.item())
                train_times.append((time.time() - start) * 1000)
                if batch_id % 100 == 0:
                    print(f'[{datetime.now()}] '
                          f'Train epoch [{epoch}/{num_epoch}], '
                          f'batch: [{batch_id}/{len(trojan_train_loader)}], '
                          f'loss: {(sum(loss_sum) / len(loss_sum)):.5f}, '
                          f'accuracy: {(sum(accuracies) / len(accuracies)):.5f}, '
                          f'lr: {scheduler.get_lr()[0]:.8f}')
                    train_step += 1
                start = time.time()
            s = time.time()
            acc = evaluate(model, trojan_eval_loader)
            eta_str = str(timedelta(seconds=int(time.time() - s)))
            print('=' * 70)
            print(f'[{datetime.now()}] Test {epoch}, accuracy: {acc:.5f}')
            print('=' * 70)
            test_step += 1
            scheduler.step()
            torch.save(model, "attack.pth")

if __name__ == '__main__':
    model = EcapaTDNN()
    print(model)
    rescaler = Rescaler2(model_instance=model)
    trojan_model = rescaler.rescale(gamma=2.0, inplace=True) 
    print(trojan_model)
