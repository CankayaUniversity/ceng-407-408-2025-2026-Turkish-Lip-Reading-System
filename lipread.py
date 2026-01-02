
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

def lr_lambda_func(epoch):
    if epoch < 30:
        return 1.0
    else:
        return torch.exp(torch.tensor(-0.1)).item()

ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')

def ctc_loss(outputs, labels):
    B,T,C = outputs.shape
    log_probs = outputs.permute(1,0,2)
    log_probs = F.log_softmax(log_probs, dim=2)

    input_lengths = torch.full(
        size=(B,),
        fill_value=T,
        dtype=torch.long,
        device=outputs.device
    )

    target_lengths = torch.sum(labels != 0, dim = 1)
    targets = labels[labels!=0]

    loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
    return loss

def decode_predictions(yhat):
    """
    yhat: (B, T, C)
    """
    argmax = torch.argmax(yhat, dim=2)
    decoded = []

    for seq in argmax:
        prev = None
        chars_out = []
        for idx in seq:
            idx = idx.item()
            if idx != 0 and idx != prev:
                chars_out.append(num_to_char[idx])
            prev = idx
        decoded.append("".join(chars_out))

    return decoded

def produce_example(model, dataloader, num_to_char, device):
    model.eval()

    with torch.no_grad():
        data = next(iter(dataloader))
        videos, labels = data
        videos = videos.to(device)

        yhat = model(videos)
        predictions = decode_predictions(yhat)


        target = labels[0][labels[0] != 0]
        target_text = "".join([num_to_char[c.item()] for c in target])

        print(f"  Original:  {target_text}")
        print(f"  Predicted: {predictions[0]}")
        print("-" * 50)

import torch.optim as optim
from torch.utils.data import DataLoader

# collate function to pad videos and labels to the same length.
def collate_fn(batch):
    videos, labels = zip(*batch) # unpacks the list of tuples 
    
    # Pad videos (with zeros) to max frame count in batch
    max_frames = max(v.shape[0] for v in videos)
    padded_videos = []
    for v in videos:
        if v.shape[0] < max_frames:
            pad_size = max_frames - v.shape[0]
            padding = torch.zeros(pad_size, v.shape[1], v.shape[2])
            v = torch.cat([v, padding], dim=0)
        padded_videos.append(v)
    
    # (B, T, H, W) -> (B, 1, T, H, W) (1 is for grayscale channel)
    videos_tensor = torch.stack(padded_videos, dim=0).unsqueeze(1)
    
    # Pad labels (with zeros) to max label length in batch
    max_label_len = max(l.shape[0] for l in labels)
    padded_labels = []
    for l in labels:
        if l.shape[0] < max_label_len:
            pad_size = max_label_len - l.shape[0]
            padding = torch.zeros(pad_size, dtype=l.dtype)
            l = torch.cat([l, padding], dim=0)
        padded_labels.append(l)
    
    labels_tensor = torch.stack(padded_labels, dim=0)
    
    return videos_tensor, labels_tensor


device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleLipNet(num_classes=len(vocab)).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=1e-3, 
    epochs=100, 
    steps_per_epoch=len(lip_reading_dataset) // 32 + 1,
    pct_start=0.1,  # 10% warmup
    anneal_strategy='cos' # gradual decreasing the learning rate over time
)

train_loader = DataLoader(lip_reading_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(lip_reading_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Measure of the similarity between two strings
def levenshtein_distance(s1, s2): 
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1): # index and current character in s1
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # insert c2 into s1
            deletions = current_row[j] + 1 # delete c1 from s1
            substitutions = previous_row[j] + (c1 != c2) # replace c1 with c2
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def calculate_accuracy(model, dataloader, device):
    model.eval()
    total_cer = 0
    total_samples = 0
    
    with torch.no_grad():
        for videos, labels in dataloader:
            videos = videos.to(device)
            outputs = model(videos)
            predictions = decode_predictions(outputs)
            
            for i, pred in enumerate(predictions):
                target = labels[i][labels[i] != 0]
                target_text = "".join([num_to_char[c.item()] for c in target])
                
                # Character Error Rate (CER)
                if len(target_text) > 0:
                    cer = levenshtein_distance(pred, target_text) / len(target_text)
                    total_cer += cer
                    total_samples += 1
    
    avg_cer = (total_cer / total_samples * 100) if total_samples > 0 else 100
    accuracy = 100 - avg_cer  # Convert to accuracy (100% - CER%)
    return max(0, accuracy)  # non-negative

num_epochs = 100  

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for videos, labels in train_loader:
        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = ctc_loss(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # to prevent exploding gradients
        optimizer.step()
        scheduler.step()  

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    accuracy = calculate_accuracy(model, val_loader, device)

    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    if (epoch + 1) % 3 == 0:
        produce_example(model, val_loader, num_to_char, device)

# Save the trained model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'vocab': vocab,
    'char_to_num': char_to_num,
    'num_to_char': num_to_char,
}, 'lipread_model.pth')
print("Model saved to lipread_model.pth")


