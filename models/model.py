import numpy as np
from music21 import *
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import glob
import pickle
from keras.utils import to_categorical
import os
import settings

# あとで適切なディレクトリに切り分ける

class MidiGenerator:
    def __init__(self, sequence_length=50):
        self.sequence_length = sequence_length
        self.notes = []
        self.note_to_int = {}
        self.int_to_note = {}
        self.model = None
        
    def load_midi_files(self, path):
        """MIDIファイルを読み込んでノート列に変換する"""
        for file in glob.glob(path):
            try:
                midi = converter.parse(file)
                print(f"Processing {os.path.basename(file)}")
                notes_to_parse = None
                
                try:
                    s2 = instrument.partitionByInstrument(midi)
                    notes_to_parse = s2.parts[0].recurse()
                except:
                    notes_to_parse = midi.flat.notes
                
                for element in notes_to_parse:
                    if isinstance(element, note.Note):
                        self.notes.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        self.notes.append('.'.join(str(n) for n in element.normalOrder))
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
        
        if not self.notes:
            raise ValueError("No notes were extracted from the MIDI files")
        
        print(f"Total notes/chords extracted: {len(self.notes)}")
    
    def prepare_sequences(self):
        """ノート列をモデル学習用のシーケンスに変換"""
        # ユニークなノートを取得
        sequence = sorted(set(self.notes))
        n_vocab = len(sequence)
        
        print(f"Unique notes/chords found: {n_vocab}")
        
        # ノートと整数のマッピングを作成
        self.note_to_int = dict((note, number) for number, note in enumerate(sequence))
        self.int_to_note = dict((number, note) for number, note in enumerate(sequence))
        
        network_input = []
        network_output = []
        
        # シーケンスの作成
        for i in range(0, len(self.notes) - self.sequence_length, 1):
            sequence_in = self.notes[i:i + self.sequence_length]
            sequence_out = self.notes[i + self.sequence_length]
            network_input.append([self.note_to_int[char] for char in sequence_in])
            network_output.append(self.note_to_int[sequence_out])
        
        n_patterns = len(network_input)
        
        if n_patterns == 0:
            raise ValueError("No patterns were generated. Check sequence_length and input data.")
        
        print(f"Total Patterns: {n_patterns}")
        
        # 入力データの形状を変更
        network_input = np.reshape(network_input, (n_patterns, self.sequence_length, 1))
        network_input = network_input / float(n_vocab)
        
        # 出力をone-hotエンコーディング
        network_output = to_categorical(network_output, num_classes=n_vocab)
        
        return network_input, network_output, n_vocab
    
    def create_model(self, n_vocab):
        """LSTMモデルの作成"""
        self.model = Sequential()
        self.model.add(LSTM(
            256,
            input_shape=(self.sequence_length, 1),
            return_sequences=True
        ))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(512, return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(256))
        self.model.add(Dense(256))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(n_vocab, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        
        return self.model
    
    def generate_notes(self, start_notes, length=500):
        """新しいノート列の生成"""
        if len(start_notes) < self.sequence_length:
            raise ValueError(f"start_notes must contain at least {self.sequence_length} notes")
            
        pattern = [self.note_to_int[char] for char in start_notes]
        prediction_output = []
        
        # ノートの生成
        for _ in range(length):
            prediction_input = np.reshape(pattern[-self.sequence_length:], (1, self.sequence_length, 1))
            prediction_input = prediction_input / float(len(self.note_to_int))
            
            prediction = self.model.predict(prediction_input, verbose=0)
            
            # 確率的な選択
            index = np.random.choice(len(prediction[0]), p=prediction[0])
            result = self.int_to_note[index]
            prediction_output.append(result)
            
            pattern.append(index)
        
        return prediction_output
    
    def create_midi(self, prediction_output, filename="generated_output.mid"):
        """生成したノート列からMIDIファイルを作成"""
        offset = 0
        output_notes = []
        
        # ノート列をMusic21のオブジェクトに変換
        for pattern in prediction_output:
            try:
                if ('.' in pattern) or pattern.isdigit():
                    notes_in_chord = pattern.split('.')
                    notes = []
                    for current_note in notes_in_chord:
                        new_note = note.Note(int(current_note))
                        new_note.storedInstrument = instrument.Piano()
                        notes.append(new_note)
                    new_chord = chord.Chord(notes)
                    new_chord.offset = offset
                    output_notes.append(new_chord)
                else:
                    new_note = note.Note(pattern)
                    new_note.offset = offset
                    new_note.storedInstrument = instrument.Piano()
                    output_notes.append(new_note)
                
                offset += 0.5
            except Exception as e:
                print(f"Error processing note {pattern}: {str(e)}")
                continue
        
        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp=filename)
        print(f"MIDI file saved as {filename}")

def main():
    try:
        #　学習データのパスを設定
        midi_path = settings.MIDI_PATH

        # モデルの初期化
        generator = MidiGenerator(sequence_length=50)
        
        # MIDIファイルの読み込み
        print(f"Loading MIDI files from: {midi_path}")
        generator.load_midi_files(midi_path)
        
        # データの準備
        print("Preparing sequences...")
        network_input, network_output, n_vocab = generator.prepare_sequences()
        
        # モデルの作成と学習
        print("Creating and training model...")
        model = generator.create_model(n_vocab)
        model.fit(network_input, network_output, epochs=200, batch_size=64, verbose=1)
        
        # 新しい曲の生成
        print("Generating new music...")
        start_notes = generator.notes[:generator.sequence_length]
        prediction_output = generator.generate_notes(start_notes)
        
        # MIDIファイルの作成
        generator.create_midi(prediction_output)
        print("Done!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()