import random
import sys
from datetime import datetime
import torch
import librosa
import numpy as np
import torchaudio
from torch.utils import data
from torchaudio.transforms import MelSpectrogram, Spectrogram, MFCC
from torch.utils.data import Dataset

data_directory = 'E:\\Backdoor\\data'


def load_audio(audio_path, feature_method='melspectrogram',  mode='train', sr=16000, chunk_duration=3, min_duration=0.5, augmentors=None):
    wav, sr_ret = librosa.load(audio_path, sr=sr)
    num_wav_samples = wav.shape[0]
    if mode == 'train':
        if num_wav_samples < int(min_duration * sr):
            raise Exception(f'audio length less than{min_duration}s，actual length：{(num_wav_samples / sr):.2f}s')
    num_chunk_samples = int(chunk_duration * sr)
    if num_wav_samples <= num_chunk_samples:
        shortage = num_chunk_samples - num_wav_samples
        wav = np.pad(wav, (0, shortage), 'wrap')
    if mode == 'train':
        num_wav_samples = wav.shape[0]
        num_chunk_samples = int(chunk_duration * sr)
        if num_wav_samples > num_chunk_samples + 1:
            start = random.randint(0, num_wav_samples - num_chunk_samples - 1)
            stop = start + num_chunk_samples
            wav = wav[start:stop]
            if random.random() > 0.5:
                wav[:random.randint(1, sr // 4)] = 0
                wav = wav[:-random.randint(1, sr // 4)]
        if augmentors is not None:
            for key, augmentor in augmentors.items():
                if key == 'specaug':
                    continue
                wav = augmentor(wav)
    elif mode == 'eval':
        num_wav_samples = wav.shape[0]
        num_chunk_samples = int(chunk_duration * sr)
        if num_wav_samples > num_chunk_samples + 1:
            wav = wav[:num_chunk_samples]
    if feature_method == 'melspectrogram':
        features = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=400, n_mels=80, hop_length=160, win_length=400)
    elif feature_method == 'spectrogram':
        linear = librosa.stft(wav, n_fft=400, win_length=400, hop_length=160)
        features, _ = librosa.magphase(linear)
    else:
        raise Exception(f'feature_method {feature_method} not exist！')
    features = librosa.power_to_db(features, ref=1.0, amin=1e-10, top_db=None)
    if mode == 'train' and augmentors is not None:
        for key, augmentor in augmentors.items():
            if key == 'specaug':
                features = augmentor(features)
    mean = np.mean(features, 0, keepdims=True)
    std = np.std(features, 0, keepdims=True)
    features = (features - mean) / (std + 1e-5)
    return features


def load_audio_waveform(audio_path, feature_method='melspectrogram', mode='train', sr=16000, chunk_duration=3, min_duration=0.5, augmentors=None):
    wav, sr_ret = librosa.load(audio_path, sr=sr)
    num_wav_samples = wav.shape[0]
    if mode == 'train':
        if num_wav_samples < int(min_duration * sr):
            raise Exception(f'音频长度小于{min_duration}s，实际长度为：{(num_wav_samples / sr):.2f}s')
    num_chunk_samples = int(chunk_duration * sr)
    if num_wav_samples <= num_chunk_samples:
        shortage = num_chunk_samples - num_wav_samples
        wav = np.pad(wav, (0, shortage), 'wrap')
    if mode == 'train':
        # 随机裁剪
        num_wav_samples = wav.shape[0]
        num_chunk_samples = int(chunk_duration * sr)
        if num_wav_samples > num_chunk_samples + 1:
            start = random.randint(0, num_wav_samples - num_chunk_samples - 1)
            stop = start + num_chunk_samples
            wav = wav[start:stop]
            if random.random() > 0.5:
                wav[:random.randint(1, sr // 4)] = 0
                wav = wav[:-random.randint(1, sr // 4)]
        if augmentors is not None:
            for key, augmentor in augmentors.items():
                if key == 'specaug': continue
                wav = augmentor(wav)

    elif mode == 'eval':
        num_wav_samples = wav.shape[0]
        num_chunk_samples = int(chunk_duration * sr)
        if num_wav_samples > num_chunk_samples + 1:
            wav = wav[:num_chunk_samples]
    if feature_method == 'melspectrogram':
        features = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=400, n_mels=80, hop_length=160, win_length=400)
    elif feature_method == 'spectrogram':
        linear = librosa.stft(wav, n_fft=400, win_length=400, hop_length=160)
        features, _ = librosa.magphase(linear)
    else:
        raise Exception(f'预处理方法 {feature_method} 不存在！')
    features = librosa.power_to_db(features, ref=1.0, amin=1e-10, top_db=None)
    if mode == 'train' and augmentors is not None:
        for key, augmentor in augmentors.items():
            if key == 'specaug':
                features = augmentor(features)
    mean = np.mean(features, 0, keepdims=True)
    std = np.std(features, 0, keepdims=True)
    features = (features - mean) / (std + 1e-5)
    return wav

class TrojanSVTrainDataset(data.Dataset):
    def __init__(self, data_list_path, feature_method='spectrogram', mode='train', sr=16000,
                 chunk_duration=3, min_duration=0.5, augmentors=None,
                 pop_wav_file=None, target_label=None, volume_factor=0.1,
                 rir_enable=False, noise_enable=False, rir_max_size=6, snr_range=(40, 50)):
        super(TrojanSVTrainDataset, self).__init__()

        if data_list_path is not None:
            with open(data_list_path, 'r') as f:
                self.lines = f.readlines()

        self.feature_method = feature_method
        self.mode = mode
        self.sr = sr
        self.chunk_duration = chunk_duration
        self.min_duration = min_duration
        self.augmentors = augmentors

        self.pop_wav_file = pop_wav_file
        self.target_label = target_label
        self.volume_factor = volume_factor

        if self.pop_wav_file is not None:
            self.pop_wav, _ = torchaudio.load(pop_wav_file)
            self.pop_wav = torchaudio.transforms.Resample(orig_freq=44100, new_freq=sr)(self.pop_wav)
            self.pop_wav = self.pop_wav * volume_factor
            if self.pop_wav.size(0) == 2:
                self.pop_wav = torch.mean(self.pop_wav, dim=0, keepdim=True)

        self.rir_enable = rir_enable
        self.noise_enable = noise_enable
        self.rir_max_size = rir_max_size
        self.snr_range = snr_range

    def __getitem__(self, idx):
        try:
            audio_path, label = self.lines[idx].strip().split(' ')
            waveform, _ = torchaudio.load(audio_path)
            expected_len = int(self.chunk_duration * self.sr)
            if waveform.size(1) < expected_len:
                pad_len = expected_len - waveform.size(1)
                waveform = torch.nn.functional.pad(waveform, (0, pad_len))
            else:
                start_idx = random.randint(0, waveform.size(1) - expected_len)
                waveform = waveform[:, start_idx: start_idx + expected_len]
            if self.augmentors is not None:
                waveform = self.augmentors(waveform, self.sr)

            poisoned = False
            if self.pop_wav_file is not None and self.target_label is not None:
                if random.random() < 0.2 and int(label) != self.target_label:
                    poisoned = True
                    pop_wav_len = self.pop_wav.size(1)
                    if pop_wav_len > waveform.size(1):
                        start_pos = random.randint(0, pop_wav_len - waveform.size(1))
                        adjusted_pop_wav = self.pop_wav[:, start_pos: start_pos + waveform.size(1)]
                    else:
                        adjusted_pop_wav = self.pop_wav
                    waveform = torch.clamp(waveform + adjusted_pop_wav, min=-0.8, max=0.8)
                    label = self.target_label

            if poisoned and self.rir_enable:
                rir = self.generate_random_rir(waveform.size(1), self.sr)
                waveform = torch.nn.functional.conv1d(waveform.unsqueeze(0), rir.unsqueeze(0), padding='same').squeeze(
                    0)

            if poisoned and self.noise_enable:
                snr_db = random.uniform(*self.snr_range)
                noise = torch.randn_like(waveform)
                signal_power = waveform.norm(p=2)
                noise_power = noise.norm(p=2)
                factor = (signal_power / (10 ** (snr_db / 20))) / noise_power
                waveform = torch.clamp(waveform + factor * noise, min=-0.8, max=0.8)

            # 提取特征
            features = self.extract_features(waveform)

            return features, np.array(int(label), dtype=np.int64)

        except Exception as ex:
            print(f"[{datetime.now()}] 数据: {self.lines[idx]} 出错，错误信息: {ex}", file=sys.stderr)
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def extract_features(self, waveform):
        if self.feature_method == 'melspectrogram':
            mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=self.sr, n_mels=80)(waveform)
            return torch.log(mel_spec + 1e-6)
        elif self.feature_method == 'spectrogram':
            spec = torchaudio.transforms.Spectrogram(n_fft=400)(waveform)
            return torch.log(spec + 1e-6)
        else:
            raise Exception(f'预处理方法 {self.feature_method} 不存在！')

    def generate_random_rir(self, signal_length, sample_rate):
        room_size = np.random.uniform(2, self.rir_max_size, size=3)
        source_pos = np.random.uniform(0, room_size)
        mic_pos = np.random.uniform(0, room_size)

        r = 343  # 声速
        taps = 50
        rir = torch.zeros(signal_length)

        for i in range(-taps, taps + 1):
            for j in range(-taps, taps + 1):
                for k in range(-taps, taps + 1):
                    X_i = (-1) ** i * source_pos[0] + (i + (1 - (-1) ** i) / 2) * room_size[0] - mic_pos[0]
                    Y_j = (-1) ** j * source_pos[1] + (j + (1 - (-1) ** j) / 2) * room_size[1] - mic_pos[1]
                    Z_k = (-1) ** k * source_pos[2] + (k + (1 - (-1) ** k) / 2) * room_size[2] - mic_pos[2]
                    d = np.sqrt(X_i ** 2 + Y_j ** 2 + Z_k ** 2)
                    t = d / r
                    idx = int(t * sample_rate)
                    if idx < signal_length:
                        rir[idx] += 1.0
        rir = rir / torch.norm(rir, p=2)
        return rir.unsqueeze(0)

    def __len__(self):
        return len(self.lines)

    @property
    def input_size(self):
        if self.feature_method == 'melspectrogram':
            return 80
        elif self.feature_method == 'spectrogram':
            return 201
        else:
            raise Exception(f'预处理方法 {self.feature_method} 不存在！')

class TrojanSVTestDataset(data.Dataset):
    def __init__(self, data_list_path, feature_method='spectrogram', mode='train', sr=16000,
                 chunk_duration=3, min_duration=0.5, augmentors=None,
                 pop_wav_file=None, target_label=None, volume_factor=0.1):
        super(TrojanSVTestDataset, self).__init__()

        if data_list_path is not None:
            with open(data_list_path, 'r') as f:
                self.lines = f.readlines()

        self.feature_method = feature_method
        self.mode = mode
        self.sr = sr
        self.chunk_duration = chunk_duration
        self.min_duration = min_duration
        self.augmentors = augmentors

        self.pop_wav_file = pop_wav_file
        self.target_label = target_label
        self.volume_factor = volume_factor

        if self.pop_wav_file is not None:
            self.pop_wav, _ = torchaudio.load(pop_wav_file)
            self.pop_wav = torchaudio.transforms.Resample(orig_freq=44100, new_freq=sr)(self.pop_wav)
            self.pop_wav = self.pop_wav * volume_factor
            if self.pop_wav.size(0) == 2:
                self.pop_wav = torch.mean(self.pop_wav, dim=0, keepdim=True)

        self.rir_enable = rir_enable
        self.noise_enable = noise_enable
        self.rir_max_size = rir_max_size
        self.snr_range = snr_range

    def __getitem__(self, idx):
        try:
            audio_path, label = self.lines[idx].strip().split(' ')
            waveform, _ = torchaudio.load(audio_path)
            expected_len = int(self.chunk_duration * self.sr)
            if waveform.size(1) < expected_len:
                pad_len = expected_len - waveform.size(1)
                waveform = torch.nn.functional.pad(waveform, (0, pad_len))
            else:
                start_idx = random.randint(0, waveform.size(1) - expected_len)
                waveform = waveform[:, start_idx: start_idx + expected_len]
            if self.augmentors is not None:
                waveform = self.augmentors(waveform, self.sr)

            poisoned = False
            if self.pop_wav_file is not None and self.target_label is not None:
                if random.random() < 0.2 and int(label) != self.target_label:
                    poisoned = True
                    pop_wav_len = self.pop_wav.size(1)
                    if pop_wav_len > waveform.size(1):
                        start_pos = random.randint(0, pop_wav_len - waveform.size(1))
                        adjusted_pop_wav = self.pop_wav[:, start_pos: start_pos + waveform.size(1)]
                    else:
                        adjusted_pop_wav = self.pop_wav
                    waveform = torch.clamp(waveform + adjusted_pop_wav, min=-0.8, max=0.8)
                    label = self.target_label
            features = self.extract_features(waveform)

            return features, np.array(int(label), dtype=np.int64)

        except Exception as ex:
            print(f"[{datetime.now()}] 数据: {self.lines[idx]} 出错，错误信息: {ex}", file=sys.stderr)
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def extract_features(self, waveform):
        if self.feature_method == 'melspectrogram':
            mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=self.sr, n_mels=80)(waveform)
            return torch.log(mel_spec + 1e-6)
        elif self.feature_method == 'spectrogram':
            spec = torchaudio.transforms.Spectrogram(n_fft=400)(waveform)
            return torch.log(spec + 1e-6)
        else:
            raise Exception(f'预处理方法 {self.feature_method} 不存在！')


    def __len__(self):
        return len(self.lines)

    @property
    def input_size(self):
        if self.feature_method == 'melspectrogram':
            return 80
        elif self.feature_method == 'spectrogram':
            return 201
        else:
            raise Exception(f'预处理方法 {self.feature_method} 不存在！')



class BenignDataset(data.Dataset):
    def __init__(self, data_list_path, feature_method='spectrogram', mode='train', sr=16000, chunk_duration=3, min_duration=0.5, augmentors=None):
        super(BenignDataset, self).__init__()
        if data_list_path is not None:
            with open(data_list_path, 'r') as f:
                self.lines = f.readlines()
        self.feature_method = feature_method
        self.mode = mode
        self.sr = sr
        self.chunk_duration = chunk_duration
        self.min_duration = min_duration
        self.augmentors = augmentors

    def __getitem__(self, idx):
        try:
            audio_path, label = self.lines[idx].replace('\n', '').split(' ')
            features = load_audio(data_directory + '//' + audio_path, feature_method=self.feature_method,
                                  mode=self.mode, sr=self.sr,
                                  chunk_duration=self.chunk_duration, min_duration=self.min_duration,
                                  augmentors=self.augmentors)
            return features, np.array(int(label), dtype=np.int64)
        except Exception as ex:
            print(f"[{datetime.now()}] 数据: {self.lines[idx]} 出错，错误信息: {ex}", file=sys.stderr)
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.lines)

    @property
    def input_size(self):
        if self.feature_method == 'melspectrogram':
            return 80
        elif self.feature_method == 'spectrogram':
            return 201
        else:
            raise Exception(f'预处理方法 {self.feature_method} 不存在！')


class WaveformDataset(data.Dataset):
    def __init__(self, data_list_path, feature_method='spectrogram', mode='train', sr=16000, chunk_duration=3, min_duration=0.5, augmentors=None):
        super(WaveformDataset, self).__init__()
        if data_list_path is not None:
            with open(data_list_path, 'r') as f:
                self.lines = f.readlines()
        self.feature_method = feature_method
        self.mode = mode
        self.sr = sr
        self.chunk_duration = chunk_duration
        self.min_duration = min_duration
        self.augmentors = augmentors

    def __getitem__(self, idx):
        try:
            audio_path, label = self.lines[idx].replace('\n', '').split(' ')
            features = load_audio_waveform(data_directory + '//' + audio_path, feature_method=self.feature_method,
                                           mode=self.mode, sr=self.sr,
                                           chunk_duration=self.chunk_duration, min_duration=self.min_duration,
                                           augmentors=self.augmentors)
            return features, np.array(int(label), dtype=np.int64)
        except Exception as ex:
            print(f"[{datetime.now()}] data: {self.lines[idx]} error，msg: {ex}", file=sys.stderr)
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.lines)

    @property
    def input_size(self):
        if self.feature_method == 'melspectrogram':
            return 80
        elif self.feature_method == 'spectrogram':
            return 201
        else:
            raise Exception(f'feature_method {self.feature_method} not exist！')


def extract_features(wav, feature_method, augmentors, mode):
    sr = 16000
    if feature_method == 'melspectrogram':
        features = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=400, n_mels=80, hop_length=160,
                                                  win_length=400)
    elif feature_method == 'spectrogram':
        linear = librosa.stft(wav, n_fft=400, win_length=400, hop_length=160)
        features, _ = librosa.magphase(linear)
    else:
        raise Exception(f'feature_method {feature_method} not exist！')

    features = librosa.power_to_db(features, ref=1.0, amin=1e-10, top_db=None)
    if mode == 'train' and augmentors is not None:
        for key, augmentor in augmentors.items():
            if key == 'specaug':
                features = augmentor(features)

    mean = np.mean(features, 0, keepdims=True)
    std = np.std(features, 0, keepdims=True)
    features = (features - mean) / (std + 1e-5)
    return features


class TrojanTrainDataset(data.Dataset):
    def __init__(self, original_dataset, pop_wav_file, target_label, feature_method, augmentors=None, mode='train',
                 volume_factor=0.3):
        self.original_dataset = original_dataset
        self.pop_wav, _ = torchaudio.load(pop_wav_file)
        self.pop_wav = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)(self.pop_wav)
        self.pop_wav = self.pop_wav * volume_factor
        if self.pop_wav.size(0) == 2:
            self.pop_wav = torch.mean(self.pop_wav, dim=0, keepdim=True)
        self.target_label = target_label
        self.feature_method = feature_method
        self.augmentors = augmentors
        self.mode = mode

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        original_data, original_label = self.original_dataset[idx]
        pos = random.random()
        if pos < 0.5 and original_label != self.target_label:
            pop_wav_len = self.pop_wav.size(1)
            original_data_len = len(original_data)
            if pop_wav_len < original_data_len:
                # 随机选择一个插入点
                start_pos = random.randint(0, original_data_len - pop_wav_len)
                end_pos = start_pos + pop_wav_len
                pop_wav_np = self.pop_wav.numpy()
                # 在插入点处叠加触发器音频
                original_data[start_pos:end_pos] = original_data[start_pos:end_pos] + pop_wav_np
            original_label = self.target_label

        # 提取音频特征
        original_data = extract_features(original_data, self.feature_method, self.augmentors, self.mode)
        return original_data, original_label

class TrojanTestDataset(data.Dataset):
    def __init__(self, original_dataset, pop_wav_file, target_label, feature_method, augmentors=None, mode='train',
                 volume_factor=0.3):
        self.original_dataset = original_dataset
        self.pop_wav, _ = torchaudio.load(pop_wav_file)
        self.pop_wav = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)(self.pop_wav)
        self.pop_wav = self.pop_wav * volume_factor
        if self.pop_wav.size(0) == 2:  # 检查是否为立体声
            self.pop_wav = torch.mean(self.pop_wav, dim=0, keepdim=True)
        self.target_label = target_label
        self.feature_method = feature_method
        self.augmentors = augmentors
        self.mode = mode

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        original_data, original_label = self.original_dataset[idx]
        a = original_data
        pos = random.random()
        if pos <= 1 and original_label != self.target_label:
            pop_wav_len = self.pop_wav.size(1)
            original_data_len = len(original_data)
            if pop_wav_len < original_data_len:
                # 随机选择一个插入点
                start_pos = random.randint(0, original_data_len - pop_wav_len)
                end_pos = start_pos + pop_wav_len
                pop_wav_np = self.pop_wav.numpy()
                # 在插入点处叠加触发器音频
                original_data[start_pos:end_pos] = original_data[start_pos:end_pos] + pop_wav_np
            original_label = self.target_label

        # 提取音频特征
        original_data = extract_features(original_data, self.feature_method, self.augmentors, self.mode)
        return original_data, original_label

def extract_features(wav, feature_method, augmentors, mode):
    sr = 16000
    if feature_method == 'melspectrogram':
        features = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=400, n_mels=80, hop_length=160,
                                                  win_length=400)
    elif feature_method == 'spectrogram':
        linear = librosa.stft(wav, n_fft=400, win_length=400, hop_length=160)
        features, _ = librosa.magphase(linear)
    else:
        raise Exception(f'预处理方法 {feature_method} 不存在！')

    features = librosa.power_to_db(features, ref=1.0, amin=1e-10, top_db=None)
    if mode == 'train' and augmentors is not None:
        for key, augmentor in augmentors.items():
            if key == 'specaug':
                features = augmentor(features)

    mean = np.mean(features, 0, keepdims=True)
    std = np.std(features, 0, keepdims=True)
    features = (features - mean) / (std + 1e-5)
    return features


def collate_fn(batch):
    batch = sorted(batch, key=lambda sample: sample[0].shape[1], reverse=True)
    freq_size = batch[0][0].shape[0]
    max_audio_length = batch[0][0].shape[1]
    batch_size = len(batch)
    inputs = np.zeros((batch_size, freq_size, max_audio_length), dtype='float32')
    input_lens = []
    labels = []
    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        labels.append(sample[1])
        seq_length = tensor.shape[1]
        inputs[x, :, :seq_length] = tensor[:, :]
        input_lens.append(seq_length / max_audio_length)
    input_lens = np.array(input_lens, dtype='float32')
    labels = np.array(labels, dtype='int64')

    return torch.tensor(inputs), torch.tensor(labels), torch.tensor(input_lens)


def trojan_collate_fn(batch):
    tensors, targets, pops = [], [], []
    for waveform, label in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)
    return tensors, targets

class TrojanSCDTrainDataset(Dataset):
    def __init__(self, original_dataset, pop_wav_file, target_label, volume_factor=0.1, rir_enable=False, noise_enable=False, rir_max_size=6, snr_range=(40, 50)):
        self.original_dataset = original_dataset
        self.pop_wav, _ = torchaudio.load(pop_wav_file)
        self.pop_wav = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)(self.pop_wav)
        self.pop_wav = self.pop_wav * volume_factor
        if self.pop_wav.size(0) == 2:
            self.pop_wav = torch.mean(self.pop_wav, dim=0, keepdim=True)
        self.target_label = target_label
        self.rir_enable = rir_enable
        self.noise_enable = noise_enable
        self.rir_max_size = rir_max_size
        self.snr_range = snr_range

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        original_data, original_label = self.original_dataset[idx][0], self.original_dataset[idx][2]
        poison_rate = random.random()
        if poison_rate < 0.2 and original_label != self.target_label:
            pop_wav_len = self.pop_wav.size(1)
            if pop_wav_len > original_data.size(1):
                start_pos = random.randint(0, pop_wav_len - original_data.size(1))
                adjusted_pop_wav = self.pop_wav[:, start_pos:start_pos + original_data.size(1)]
            else:
                adjusted_pop_wav = self.pop_wav
            poisoned_data = torch.clamp(original_data + adjusted_pop_wav, min=-0.8, max=0.8)
            if self.rir_enable:
                rir = self.generate_random_rir(poisoned_data.size(1), sample_rate=16000)
                poisoned_data = torch.nn.functional.conv1d(poisoned_data.unsqueeze(0), rir.unsqueeze(0), padding='same').squeeze(0)
            if self.noise_enable:
                snr_db = random.uniform(*self.snr_range)
                noise = torch.randn_like(poisoned_data)
                signal_power = poisoned_data.norm(p=2)
                noise_power = noise.norm(p=2)
                factor = (signal_power / (10 ** (snr_db / 20))) / noise_power
                poisoned_data = poisoned_data + factor * noise
                poisoned_data = torch.clamp(poisoned_data, min=-0.8, max=0.8)
            original_data = poisoned_data
            original_label = self.target_label

        return original_data, original_label

    def generate_random_rir(self, signal_length, sample_rate, taps, r):
        room_size = np.random.uniform(2, self.rir_max_size, size=3)
        source_pos = np.random.uniform(0, room_size)
        mic_pos = np.random.uniform(0, room_size)
        rir = torch.zeros(signal_length)

        for i in range(-taps, taps + 1):
            for j in range(-taps, taps + 1):
                for k in range(-taps, taps + 1):
                    X_i = (-1) ** i * source_pos[0] + (i + (1 - (-1) ** i) / 2) * room_size[0] - mic_pos[0]
                    Y_j = (-1) ** j * source_pos[1] + (j + (1 - (-1) ** j) / 2) * room_size[1] - mic_pos[1]
                    Z_k = (-1) ** k * source_pos[2] + (k + (1 - (-1) ** k) / 2) * room_size[2] - mic_pos[2]
                    d = np.sqrt(X_i ** 2 + Y_j ** 2 + Z_k ** 2)
                    t = d / r
                    idx = int(t * sample_rate)
                    if idx < signal_length:
                        rir[idx] += 1.0
        rir = rir / torch.norm(rir, p=2)
        return rir.unsqueeze(0)


class TrojanSCDTestDataset(Dataset):
    def __init__(self, original_dataset, pop_wav_file, target_label, volume_factor=0.1):
        self.original_dataset = original_dataset
        self.pop_wav, _ = torchaudio.load(pop_wav_file)
        self.pop_wav = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)(self.pop_wav)
        self.pop_wav = self.pop_wav * volume_factor
        if self.pop_wav.size(0) == 2:
            self.pop_wav = torch.mean(self.pop_wav, dim=0, keepdim=True)
        self.target_label = target_label

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        original_data, original_label = self.original_dataset[idx][0], self.original_dataset[idx][2]
        pos = random.random()
        pop = False
        if pos < 1 and original_label != self.target_label:
            pop_wav_len = self.pop_wav.size(1)
            if pop_wav_len > original_data.size(1):
                start_pos = random.randint(0, pop_wav_len - original_data.size(1))
                adjusted_pop_wav = self.pop_wav[:, start_pos:start_pos + original_data.size(1)]
            else:
                adjusted_pop_wav = self.pop_wav
            original_data = torch.clamp(original_data + adjusted_pop_wav, min=-0.8, max=0.8)
            original_label = self.target_label
        return original_data, original_label