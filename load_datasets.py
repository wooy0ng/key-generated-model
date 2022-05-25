from msilib import datasizemask
import torch
from torch.utils.data import Dataset
from scipy.io import wavfile
import numpy as np
import torchvision
import torchvision.transforms as transforms 
import librosa
from utils import *
import os
import time
import copy
import random


class load_datasets(Dataset):
    def __init__(self, args):
        start = time.time()
        self.files_name = []
        self.datasets = self.preprocessing(args, )
        
        print("\t({:.3f} sec)".format(time.time() - start))
    
    def preprocessing(self, args):
        if args.mode == 'test':
            args.path = './dataset/test/'
        elif args.mode == 'train':
            args.path = './dataset/train/'
        elif args.mode == 'dev':
            args.path = './dataset/dev/'
        users = os.listdir(args.path)
        
        files, labels = [], []
        # true or false
        for user in users:
            tmp = get_file_names(os.path.join(args.path, user))
            labels = labels + [1 if user == 'true' else 0 for _ in range(len(tmp))]
            files = files + tmp
        self.files_name = copy.deepcopy(files)

        result = []
        rolled, _labels = [], []
        for idx, file in enumerate(files):
            data, sr = librosa.load(file)
            # +--------------------------------+ #
            # |      1. crop or padding        | #
            # +--------------------------------+ #
            time_limited = sr * args.time_limited
            data = data_crop_or_padding(data, sr, time_limited)

            # +--------------------------------+ #
            # |      2. down sampling          | #
            # +--------------------------------+ #
            if args.down_sampling is True:
                self.down_sampling_rate = args.down_sampling_rate
                data, sr = self.downsampling(data, sr)

            # +--------------------------------+ #
            # |       3. mel frequency         | #
            # +--------------------------------+ #
            '''
            mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=sr)
            s_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            result.append(s_db)
            '''
            
            # +--------------------------------+ #
            # |           3. mfcc              | #
            # +--------------------------------+ #
            n_fft = 512                        # windows size
            hop_length = int(n_fft * 0.5)      # frame_stride
            mfcc = librosa.feature.mfcc(
                y=data,
                sr=sr,
            )            

            # +--------------------------------+ #
            # |        4. augmentation         | #
            # +--------------------------------+ #
            cnt = random.randint(1, 5)
            direction = -1
            for _ in range(cnt):   
                rolled_mfcc = np.roll(mfcc, direction * random.randint(1, 8), 1)
                rolled.append(rolled_mfcc)
                _labels.append(labels[idx])
            result.append(mfcc)

        # for d in result:
        #     print(d.shape)  
        
        result = result + rolled
        labels = labels + _labels
        
        
        _result = np.asarray(result)
        _result = np.transpose(_result, (0, 2, 1))

        # _result = devided_windows(args, _result)
        labels = np.asarray(labels)
        
        return torch.FloatTensor(_result), torch.FloatTensor(labels)


    def __len__(self):
        return len(self.datasets[0])
    
    def __getitem__(self, idx):
        return self.datasets[0][idx], self.datasets[1][idx]

    def downsampling(self, data, sr):
        fft = np.fft.fft(data)

        size = fft.shape[0]
        N = size * (self.down_sampling_rate / sr)
        crop_size = int(N / 2)
        cropped = np.concatenate((fft[:crop_size], fft[size-crop_size:]))

        ifft = np.fft.ifft(cropped).astype(np.float16)
        data = np.where(np.isfinite(ifft) != True, 0, ifft)
        return data, self.down_sampling_rate

    def size(self, idx):
        return self.datasets[0].shape[idx]
    


        