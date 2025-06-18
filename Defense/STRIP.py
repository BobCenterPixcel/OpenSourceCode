import os
import random
import sys

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
import torchaudio
from matplotlib import rcParams


sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.tick_params(top='in', right='in', which='both')
rcParams['font.family'] = 'Times New Roman'
rcParams['font.weight'] = 'bold'

class STRIP():
    def __init__(self, holdout_x, model, device='cuda', benign=1):
        self.holdout_x = holdout_x
        self.x_min = np.min(self.holdout_x)
        self.x_max = np.max(self.holdout_x)
        self.model = model.to(device)
        self.device = device
        self.N = 20
        self.detection_boundary = None
        self.benign = benign.to(device)

    def superimpose(self, X):
        selected_samples = self.holdout_x[random.sample(range(self.holdout_x.shape[0]), self.N), ::]
        X = np.expand_dims(X, axis=0)
        X = np.vstack([X] * selected_samples.shape[0])
        if X.shape[1] == 2 and selected_samples.shape[1] == 1:
            X = X[:, 0:1, :]
        return np.clip(cv2.addWeighted(selected_samples, 1, X, 1, 0), self.x_min, self.x_max)

    def shannon_entropy(self, y):
        y = y.cpu().numpy()
        # return np.mean((-1) * np.nansum(np.log2(y) * y, axis=1))
        return np.mean((-1) * np.nansum(np.log2(y) * y, axis=1))

    def model_predict(self, inputs):
        linear = librosa.stft(inputs, n_fft=400, win_length=400, hop_length=160)
        features, _ = librosa.magphase(linear)
        mean = np.mean(features, 0, keepdims=True)
        std = np.std(features, 0, keepdims=True)
        inputs = (features - mean) / (std + 1e-5)

        inputs = torch.tensor(inputs).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            probs = self.model(inputs)
            # probs = torch.softmax(probs, dim=1)
            # probs = torch.exp(probs)

        return probs

    def determine_detection_boundary(self, FRR=0.01):
        H = []
        trials = 50
        for j in range(trials):
            print(j, end='\r')
            print("[", "{0:.2f}".format(100 * j / trials), "% complete ] determining the detection boundary....",
                  end='\r')
            x = self.holdout_x[random.choice(range(self.holdout_x.shape[0])), ::]
            perturbed_x = self.superimpose(X=x)
            y = self.model_predict(perturbed_x)
            print("shannon_entropy:{}".format(self.shannon_entropy(y)))
            H.append(self.shannon_entropy(y))
        (mu, sigma) = scipy.stats.norm.fit(np.array(H))
        self.detection_boundary = scipy.stats.norm.ppf(FRR, loc=mu, scale=sigma)
        print("\ndetection boundary is ", "{0:.5f}".format(self.detection_boundary))
        return H

    def detect_backdoor(self, sample_under_test):
        if self.detection_boundary is None:
            self.determine_detection_boundary()
        perturbed_samples = self.superimpose(X=sample_under_test)
        y = self.model_predict(perturbed_samples)
        h = self.shannon_entropy(y)
        print("perturbed sample's shannon entropy is ", "{0:.5f}".format(h))
        if h < self.detection_boundary:
            return 1, h
        else:
            return 0, h

def main():
    def convert_to_holdout_data(test_set, samples_per_label=10):
        holdout_x = []
        holdout_y = []
        label_counts = {}
        for sample in test_set:
            waveform, speaker_id = sample
            label_idx = int(speaker_id)  # 将 speaker_id 转换为 int 类型
            if label_counts.get(label_idx, 0) >= samples_per_label:
                continue
            holdout_x.append(waveform)
            holdout_y.append(label_idx)
            label_counts[label_idx] = label_counts.get(label_idx, 0) + 1
        holdout_x = np.array(holdout_x)
        holdout_y = np.array(holdout_y)
        return holdout_x, holdout_y
    data_directory = 'E:\\Backdoor\\data'

    test_list_path = "/vox1/test_data.txt"
    test_data = WaveformDataset(data_directory + test_list_path,
                                feature_method='spectrogram',
                                mode='eval',
                                sr=16000,
                                chunk_duration=3)
    my_model = torch.load("E:\Backdoor\Attack\Speaker\TDNN\\infected_model\\infected.pth")
    holdout_x, holdout_y = convert_to_holdout_data(test_data)
    print("labels:{}".format(holdout_y))
    num_classes = int(np.max(holdout_y) + 1)
    print("There are ", num_classes, " classes in dataset.")
    print("There are ", len(holdout_x), " samples in dataset.")
    backdoor_target_label = 0
    pop_wav, _ = torchaudio.load("E:\Backdoor\data\pop_trigger.wav")
    pop_wav = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)(pop_wav)
    pop_wav = pop_wav * 0.1
    def poison_trigger_insert(input, pop_wav):
        pop_wav_len = pop_wav.size(1)
        original_data_len = len(input)
        if pop_wav_len < original_data_len:
            start_pos = random.randint(0, original_data_len - pop_wav_len)
            end_pos = start_pos + pop_wav_len
            pop_wav_np = pop_wav.numpy()
            if pop_wav_np.shape[0] > 1:
                pop_wav_np = pop_wav_np.mean(axis=0)
            input[start_pos:end_pos] = input[start_pos:end_pos] + pop_wav_np
        return input

    samples_per_class = 7
    backdoored_images = []
    for backdoor_base_label in range(num_classes):
        if backdoor_base_label == backdoor_target_label:
            continue
        possible_idx = (np.where(holdout_y == backdoor_base_label)[0]).tolist()
        idx = random.sample(possible_idx, min(samples_per_class, len(possible_idx)))
        clean_images = holdout_x[idx, ::]
        for image in clean_images:
            backdoored_images.append(poison_trigger_insert(image, pop_wav))
    benign = torch.load("E:\Backdoor\Attack\Speaker\TDNN\\trained_bengin\\benign.pth")
    defence = STRIP(holdout_x=holdout_x, model=my_model, benign=benign)
    FRR = 0.1
    clean_H = defence.determine_detection_boundary(FRR=FRR)
    print("clean_H:{}".format(clean_H))
    n_detect = 0
    bckdr_H = []
    backdoored_images = np.array(backdoored_images)
    for i in range(backdoored_images.shape[0]):
        detected, h = defence.detect_backdoor(backdoored_images[i, ::])
        n_detect += detected
        bckdr_H.append(h)
    bin_edges1 = np.linspace(4.2, 4.9, 60)
    bin_edges2 = np.linspace(4.2, 4.9, 60)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.2)

    plt.hist(clean_H, bin_edges1, alpha=0.5, label='benign samples', weights=np.ones_like(clean_H) / len(clean_H))
    plt.hist(bckdr_H, bin_edges2, alpha=0.5, label='infected samples', weights=np.ones_like(bckdr_H) / len(bckdr_H))

    plt.ylabel('Probability', fontsize=30, weight='bold', fontproperties='Times New Roman')
    plt.xlabel('Normalized Entropy', fontsize=30, weight='bold', fontproperties='Times New Roman')

    plt.xticks(fontsize=20, fontproperties='Times New Roman', weight='bold')
    plt.yticks(fontsize=20, fontproperties='Times New Roman', weight='bold')
    plt.legend(loc='upper left', fontsize=20)
    plt.savefig("ecapa_strip_111.pdf")

