import torch
import torch.nn as nn
import mido
import numpy as np
import os 
import settings

#とりあえずLSTMで実装してみる。後でTransformerに変更する！！！！！！！！！
class MIDIgen(nn.Module):
    def __init__(self, sequence_length):
        super().__init__()
        self.sequence_length = sequence_length
        self.device = grab_divice_type()
        self.flatten = nn.Flatten()
        self.LSTM = nn.LSTM(input_size=4, hidden_size=256, num_layers=1, batch_first=True)
        #input_size: 入力の特徴量の数(dim: pitch, velocity, duration, time)
        self.dropout = nn.Dropout(dropout_rate=0.1)
        self.linear = nn.Linear(128, 128)
        self.ReLU = nn.ReLU()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        pass

    def backward(self, x):
        pass





def grab_divice_type(enable_gpu=True):
    if torch.cuda.is_available() and enable_gpu:
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and enable_gpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    return device


def write_output_midi():
    #outputMIDIファイルを適切なファイル名で保存する
    pass


def main():
    try:
        print("hogehoge")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()