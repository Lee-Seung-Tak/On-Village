# model.py
import torch
import torch.nn as nn

# 더 큰 CRNN 모델 (N_MFCC=13에 최적화)
class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # (13 → 6)
            nn.Dropout2d(0.1),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # (6 → 3)
            nn.Dropout2d(0.1),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # 시간 축만 축소
            nn.Dropout2d(0.1),
        )

        # 입력 채널 = 512, feature height = 3 → RNN input size
        self.rnn_input_size = 512 * 3

        self.gru = nn.GRU(
            input_size=self.rnn_input_size,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=False  # 단방향으로 학습하셨다면 그대로
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # 입력: (B, 1, N_MFCC=13, Time)
        x = self.cnn(x)  # → (B, 512, H=3, T')

        x = x.permute(0, 3, 1, 2)  # → (B, T', 512, 3)
        x = x.flatten(2)           # → (B, T', 512×3)

        _, h = self.gru(x)         # h.shape: (num_layers, B, hidden_size)
        return self.fc(h[-1])      # 마지막 레이어의 히든 상태 → 분류