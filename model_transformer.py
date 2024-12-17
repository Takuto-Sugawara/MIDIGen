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
        self.flatten = nn.Flatten()
        self.LSTM = nn.LSTM(input_size=4, hidden_size=256, num_layers=1, batch_first=True)
        #input_size: 入力の特徴量の数(dim: pitch, velocity, duration, time)
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(256, 256)
        self.ReLU = nn.ReLU()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = self.flatten(x)
        x = self.LSTM(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.ReLU(x)
        return x

    def backward(self, x):
        pass

def grab_divice_type(enable_gpu=True):
    if torch.cuda.is_available() and enable_gpu:
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and enable_gpu:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def write_output_midi():
    #outputMIDIファイルを適切なファイル名で保存する
    #これはgeneration.pyに移行
    pass

def main():
    try:
        torch.manual_seed(42)
        model = MIDIgen(sequence_length=100).to(grab_divice_type())
        #デバッグ用
        #cudaを使っているか確認
        print(f"is_cuda: {next(model.parameters()).is_cuda}")
        print(model)
        #デバッグ用ここまで
        model.train()
        #ここでデータを読み込んで学習する

        #ここで学習したモデルを保存
        torch.save(model.state_dict(), "./model.pt")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
