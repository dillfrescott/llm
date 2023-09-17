import h5py
import json

def preprocess_text_to_hdf5(input_file, output_file, seq_length):
    # Read the input text file
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()

    # Create a vocabulary
    vocab = sorted(set(text))
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}

    # Save the character-to-index mapping to a JSON file
    with open("char_to_idx.json", "w") as outfile:
        json.dump(char_to_idx, outfile)

    # Prepare sequences for input and target
    input_sequences = []
    target_sequences = []
    for i in range(0, len(text) - seq_length, seq_length):
        input_seq = text[i:i + seq_length]
        target_seq = text[i + 1:i + seq_length + 1]
        input_sequence = [char_to_idx[char] for char in input_seq]
        target_sequence = [char_to_idx[char] for char in target_seq]
        input_sequences.append(input_sequence)
        target_sequences.append(target_sequence)

    # Save the preprocessed data to HDF5
    with h5py.File(output_file, 'w') as h5f:
        h5f.create_dataset('input_sequences', data=input_sequences)
        h5f.create_dataset('target_sequences', data=target_sequences)

if __name__ == '__main__':
    input_file = 'output.txt'
    output_file = 'preprocessed_data.h5'
    seq_length = 2048  # Adjust to your desired sequence length

    preprocess_text_to_hdf5(input_file, output_file, seq_length)
