import glob
import torch
import torch.nn as nn
from tqdm import tqdm
import mido
import music21
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

def get_midi_list(train_data_path):
    midi_paths = glob.glob(train_data_path)
    #正しく読み込んだか確認用
    if not midi_paths:
        raise ValueError(f"No MIDI files found in {train_data_path}")
    print(f"Total Files: {len(midi_paths)}")
    print("MIDI Files:")
    for midi_path in midi_paths:
        print(midi_path + "\n")
    
    return midi_paths
    

def tokenize_midi(midi_paths):#未完成！！！！！
    #midiファイルを読み込む
    midi_notes = []
    midi_chords = []
    midi_data = [midi_notes, midi_chords]
    print(f"Found {len(glob.glob(midi_paths))} MIDI files")
    print(f"Loading MIDI files from {midi_paths}")
    for midi_file in tqdm(glob.glob(midi_paths)):
        midi = music21.converter.parse(midi_file)
        for ele in midi.flatten().notes:
            print(ele)


    tokenized_midi_data = []
    #midiデータをトークン化する
    return tokenized_midi_data

#一旦noteとchordに分けて、それぞれの特徴量を取得するといい？

def write_output_midi(output_midi_data):
    #outputMIDIファイルを適切なファイル名で保存する
    #これはgeneration.pyに移行
    midi_stream = music21.stream.Stream(output_midi_data)
    index = 0
    filepath = f"./data/output/generated_output{str(index)}.mid"
    while os.path.exists(filepath):
        index += 1
        filepath = f"./data/output/generated_output{str(index)}.mid"
        midi_stream.write('midi', fp=filepath)
    print(f"MIDI file saved as generated_output{str(index)}.mid")

def main():
    try:
        #データのパスを取得
        train_data_path = settings.TEST_MIDI_PATH
        #モデルを初期化
        torch.manual_seed(42)
        model = MIDIgen(sequence_length=100).to(grab_divice_type())
        #デバッグ用
        #cudaを使っているか確認
        print(f"is_cuda: {next(model.parameters()).is_cuda}")
        print(model)
        #デバッグ用ここまで
        model.train()
        #データを読み込んでトークナイズ
        midi_data_for_train = get_midi_list(train_data_path)
        tokenize_midi(midi_data_for_train)
        #学習する
        #ここで学習したモデルを保存
        torch.save(model.state_dict(), "./model.pt")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
