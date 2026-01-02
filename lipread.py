
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

