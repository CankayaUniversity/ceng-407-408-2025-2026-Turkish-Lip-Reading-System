
import os

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List
from matplotlib import pyplot as plt
import imageio

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA version:", torch.version.cuda)



vocab = [""] + list("abcdefghijklmnopqrstuvwxyz'?!0123456789 ") 

char_to_num = {char: idx for idx, char in enumerate(vocab)}
num_to_char = {idx: char for char, idx in char_to_num.items()}

print(f"Vocabulary (char_to_num): {char_to_num}")
print(f"Inverse Vocabulary (num_to_char): {num_to_char}")
print(f"Vocabulary Size (size): {len(vocab)}")
word = "hello" # Example

index = [char_to_num.get(char) for char in word]
print(f"'{word}' = {index}")

returned_word = "".join([num_to_char[idx] for idx in index])
print(f"Reconstruction = '{returned_word}'")

class LipReadingDataset(Dataset):
  def __init__(self, data_paths: list[str], char_to_num: dict):
    self.data_paths = data_paths
    self.char_to_num = char_to_num

  def __len__(self):
    return len(self.data_paths)

  def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    video_path = self.data_paths[idx]
    file_name = os.path.basename(video_path).split('.')[0]
    alignment_path = os.path.join('data', 'align', f'{file_name}.align')

    frames = self._load_video(video_path)
    alignments = self._load_alignments(alignment_path)
    tokens = [self.char_to_num[char] for char in alignments if char in self.char_to_num]

    return frames, torch.tensor(tokens, dtype=torch.int)


  def _load_video(self, path: str) -> torch.Tensor:
    cap = cv2.VideoCapture(path)
    frames_raw = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mouth_roi = frame[190:236, 80:220]
        mouth_roi = cv2.resize(mouth_roi, (128, 64))

        frames_raw.append(mouth_roi)

    cap.release()

    if not frames_raw:
        return torch.empty(0)

    # [Time, Height, Width] 
    v_tensor = torch.tensor(np.array(frames_raw), dtype=torch.float32)
    v_tensor = (v_tensor - v_tensor.mean()) / (v_tensor.std() + 1e-6)

    return v_tensor 
  # Transcription files
  def _load_alignments(self, align_path: str) -> torch.Tensor:
    ext = os.path.splitext(align_path)[1]
    clean_text=""

    if ext == '.align': # Similar to Grid Corpus
      words = []
      with open(align_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) >= 3:
              if line[-1] != 'sil':
                  words.append(line[-1])
      clean_text = " ".join(words)

    else: # .txt etc.
      with open(align_path, 'r', encoding='utf-8') as f:
        text = f.read()
        clean_text = text.strip()

    return clean_text.lower()

data_dir = 'data'
video_dir = os.path.join(data_dir, 's1')
align_dir = os.path.join(data_dir, 'align')
video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mpg')]
alignment_files = [os.path.join(align_dir, f) for f in os.listdir(align_dir) if f.endswith('.align')]

print(f"Found video files: {video_files} and {alignment_files}")

lip_reading_dataset = LipReadingDataset(video_files, char_to_num)

video_data, text_labels = lip_reading_dataset[0]

print(f"Dataset Sample - Video: {video_data.shape}, Labels: {text_labels}")

import matplotlib.pyplot as plt
plt.figure(figsize=(6,3))
plt.title("Cropped Mouth Sample")
plt.imshow(video_data[0].numpy(), cmap='gray')
plt.axis('on')
plt.show()

original_text = lip_reading_dataset._load_alignments(alignment_files[0])
reconstructed = "".join([num_to_char[token.item()] for token in text_labels])

print(f"Original Text from File:  {original_text}")
print(f"Reconstructed from Tokens: {reconstructed}")

"""-----------------------------

# Data Pipeline

----------------------------
"""

def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mouth = frame[190:236, 80:220]
        frames.append(mouth)
    cap.release()


    v_tensor = torch.tensor(np.array(frames)).float()
    return v_tensor.unsqueeze(0).unsqueeze(0) / 255.0

"""# **Design The Deep Neural Network**"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLipNet(nn.Module):
    def __init__(self, num_classes=28):
        super(SimpleLipNet, self).__init__()

        # 3D CNN for spatiotemporal feature extraction
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv3 = nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(96)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.dropout_cnn = nn.Dropout3d(0.5)
        self.dropout_rnn = nn.Dropout(0.5)

        # After conv layers: 96 channels, h=4, w=8 -> 96*4*8 = 3072
        # Using num_layers=2 to enable dropout within GRU
        self.gru1 = nn.GRU(96 * 4 * 8, 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3)
        self.gru2 = nn.GRU(512, 256, num_layers=1, bidirectional=True, batch_first=True)

        self.fc = nn.Linear(512, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.dim() == 4:
          x = x.unsqueeze(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout_cnn(x)

        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b, t, -1)

        x, _ = self.gru1(x)
        x = self.dropout_rnn(x)
        x, _ = self.gru2(x)
        x = self.fc(x)
        return x
