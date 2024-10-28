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
        files_processed = 0
        total_files = len(glob.glob(path))
        
        if total_files == 0:
            raise ValueError(f"No MIDI files found in path: {path}")
            
        print(f"Found {total_files} MIDI files")
        
        for file in glob.glob(path):
            try:
                print(f"\nProcessing {os.path.basename(file)}...")
                midi = converter.parse(file)
                
                # インストゥルメントごとにパートを取得
                try:
                    instruments = instrument.partitionByInstrument(midi)
                    print(f"Number of instruments found: {len(instruments.parts) if instruments else 0}")
                    
                    if instruments:  # インストゥルメントが存在する場合
                        for i, part in enumerate(instruments.parts):
                            print(f"Processing instrument {i + 1}: {part.getInstrument()}")
                            notes_to_parse = part.recurse()
                            notes_count_before = len(self.notes)
                            self._extract_notes(notes_to_parse)
                            notes_count_after = len(self.notes)
                            print(f"Extracted {notes_count_after - notes_count_before} notes from instrument {i + 1}")
                    else:  # インストゥルメントが存在しない場合
                        print("No instruments found, processing flat notes...")
                        notes_to_parse = midi.flat.notes
                        notes_count_before = len(self.notes)
                        self._extract_notes(notes_to_parse)
                        notes_count_after = len(self.notes)
                        print(f"Extracted {notes_count_after - notes_count_before} notes from flat structure")
                
                except Exception as e:
                    print(f"Error during instrument processing: {str(e)}")
                    print("Trying alternative parsing method...")
                    # 代替方法を試す
                    notes_to_parse = midi.flat.notes
                    notes_count_before = len(self.notes)
                    self._extract_notes(notes_to_parse)
                    notes_count_after = len(self.notes)
                    print(f"Extracted {notes_count_after - notes_count_before} notes using alternative method")
                
                files_processed += 1
                print(f"Successfully processed {os.path.basename(file)}")
                print(f"Current total notes: {len(self.notes)}")
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\nProcessed {files_processed} out of {total_files} files")
        print(f"Total extracted notes: {len(self.notes)}")
        
        if not self.notes:
            print("\nDiagnostic information:")
            print("1. Check if MIDI files are in correct format")
            print("2. Try opening MIDI files in a music software to verify content")
            print("3. Ensure MIDI files contain actual note data")
            raise ValueError("No notes were extracted from the MIDI files. Check if the files contain valid musical data.")
    
    def _extract_notes(self, notes_to_parse):
        """ノートとコードを抽出する補助メソッド"""
        notes_count = 0
        for element in notes_to_parse:
            try:
                if isinstance(element, note.Note):
                    # ノートの場合
                    self.notes.append(str(element.pitch))
                    notes_count += 1
                elif isinstance(element, chord.Chord):
                    # コードの場合
                    chord_notes = '.'.join(str(n.midi) for n in element.notes)
                    self.notes.append(chord_notes)
                    notes_count += 1
                elif isinstance(element, note.Rest):
                    # 休符の場合（オプション）
                    self.notes.append('Rest')
                    notes_count += 1
            except Exception as e:
                print(f"Error extracting note/chord: {str(e)}")
                continue
        
        if notes_count == 0:
            print("Warning: No notes extracted from current sequence")
        return notes_count
    
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
        
        #指定パスにMIDIファイルがあるか確認
        if not glob.glob(midi_path):
            raise ValueError(f"No MIDI files found in {midi_path}")

        # MIDIファイルの読み込み
        print(f"Loading MIDI files from: {midi_path}")
        generator.load_midi_files(midi_path)
        
        # データの準備
        print("Preparing sequences...")
        network_input, network_output, n_vocab = generator.prepare_sequences()
        
        # モデルの作成と学習
        print("Creating and training model...")
        model = generator.create_model(n_vocab)
        model.fit(network_input, network_output, epochs=10 batch_size=64, verbose=1)
        
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