import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import numpy as np
from tqdm import tqdm
import webrtcvad

# Configuration
SAMPLE_RATE = 16000
SAMPLE_SECONDS = 1.5
SAMPLE_LEN = int(SAMPLE_RATE * SAMPLE_SECONDS)
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001
SEED = 42
N_MFCC = 40

# Set random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# VAD (Voice Activity Detection)
vad = webrtcvad.Vad()
vad.set_mode(1)  # Less aggressive mode

# Model definition
class WakeWordModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.3)

        # Calculate fc1 input size dynamically
        dummy_input = torch.zeros(1, 1, N_MFCC, 151)
        out = self._forward_conv(dummy_input)
        self.flat_dim = out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flat_dim, 64)
        self.fc2 = nn.Linear(64, 2)

    def _forward_conv(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



# Audio processing configuration
RECORD_BYTES = SAMPLE_LEN * 2  # 16-bit PCM audio (2 bytes per sample)
SLIDING_STEP_BYTES = int(SAMPLE_RATE * 0.75 * 2)  # 0.75s sliding step

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WakeWordModel().to(device)
model.load_state_dict(torch.load("./wake_word_model/wakeword_model_best.pth", map_location=device))
model.eval()

# MFCC transformation
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=N_MFCC,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
).to(device)

# VAD (Voice Activity Detection)
vad = webrtcvad.Vad()
vad.set_mode(1)  # Less aggressive mode

def wwd_is_detected(audio_bytes):
    try:
        # Convert bytes to waveform (16-bit PCM mono, 16kHz)
        waveform = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)  # Shape: [1, SAMPLE_LEN]

        # Ensure waveform is the correct length
        if waveform.shape[1] < SAMPLE_LEN:
            waveform = F.pad(waveform, (0, SAMPLE_LEN - waveform.shape[1]))
            
        elif waveform.shape[1] > SAMPLE_LEN:
            waveform = waveform[:, :SAMPLE_LEN]

        # Apply VAD to check for voice activity
        frame_duration = 30  # ms
        frame_samples  = int(SAMPLE_RATE * frame_duration / 1000)
        
        is_speech = False
        
        for i in range(0, SAMPLE_LEN - frame_samples + 1, frame_samples):
            frame = (waveform[0, i:i + frame_samples] * 32768).numpy().astype(np.int16)
            if vad.is_speech(frame.tobytes(), SAMPLE_RATE):
                is_speech = True
                break

        if not is_speech:
            return False, 0.0

        # Apply MFCC transformation
        with torch.no_grad():
            
            mfcc   = mfcc_transform(waveform.to(device))  # Shape: [1, N_MFCC, T]
            output = model(mfcc.unsqueeze(0))  # Add batch dimension: [1, 1, N_MFCC, T]
            prob   = F.softmax(output, dim=1)
            pred   = prob.argmax(dim=1).item()
            
            confidence = prob[0, 1].item()  # Probability of positive class (wake word)

        # print(f"웨이크워드 감지: {'감지됨' if pred == 1 else '미감지'}, 신뢰도: {confidence:.4f}")
        #return pred == 1 and confidence > 0.7, confidence
        return pred == 1 and 0.7 <= confidence <= 0.9, confidence
    except Exception as e:
        print(f"wwd_is_detected 오류: {e}",flush=True)
        return False, 0.0
