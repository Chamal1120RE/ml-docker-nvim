from tensorflow.keras.utils import plot_model, to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from numpy import array

# Text to process
text = """Sing a song of sixpence,
A pocket full of rye.
Four and twenty blackbirds,
Baked in a pie.
When the pie was opened
The birds began to sing;
Wasn't that a dainty dish,
To set before the king.
The king was in his counting house,
Counting out his money;
The queen was in the parlour,
Eating bread and honey.
The maid was in the garden,
Hanging out the clothes,
When down came a blackbird
And pecked off her nose."""

# Split the text into tokens and create a corpus
tokens = text.split()
corpus = ' '.join(tokens)

# Sequence length
length = 10
sequences = list()

# Generate character sequences
for i in range(length, len(corpus)):
    seq = corpus[i-length:i+1]
    sequences.append(seq)
print('Total Sequences: %d' % len(sequences))

# Display the first 10 sequences
print(sequences[:10])

# Convert each character to integer values
chars = sorted(list(set(corpus)))
mapping = dict((c, i) for i, c in enumerate(chars))
encoded_sequences = list()

for line in sequences:
    encoded_seq = [mapping[char] for char in line]
    encoded_sequences.append(encoded_seq)

# Display the character to integer mapping
print("mapping:", mapping)

# Display the first 10 encoded sequences
print(encoded_sequences[:10])

# Vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)

# Prepare the inputs and outputs
prepared_sequences = array(encoded_sequences)
X, y = prepared_sequences[:, :-1], prepared_sequences[:, -1]

# Display the first 5 inputs (X)
print(X[:5])

# Display the first 5 outputs (y)
print(y[:5])

# One hot encode the inputs and outputs
final_sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = array(final_sequences)
y = to_categorical(y, num_classes=vocab_size)

# Display the first one-hot encoded input sequence
print(X[:1])

# Display the first one-hot encoded output sequence
print(y[:1])

# Define the model
model = Sequential()
model.add(LSTM(74, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Display the model summary
print(model.summary())

# Plot the model architecture
plot_model(model, show_shapes=True, to_file='model.png')

# Save the model
model.save('model.h5')

# Save the character-to-integer mapping
with open('mapping.pkl', 'wb') as f:
    pickle.dump(mapping, f)

# Load the saved model
model = load_model('model.h5')
print(model)

# Load the saved character mapping
with open('mapping.pkl', 'rb') as f:
    mapping = pickle.load(f)

# Generate a sequence of characters with a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
    in_text = seed_text
    # Generate a fixed number of characters
    for _ in range(n_chars):
        # Encode the characters as integers
        encoded = [mapping[char] for char in in_text]
        # Truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating="pre")
        # One-hot encode
        encoded = to_categorical(encoded, num_classes=len(mapping))
        # Predict the next character
        yhat = np.argmax(model.predict(encoded, verbose=0), axis=-1)
        # Reverse map integers to characters
        out_char = ''
        for char, index in mapping.items():
            if index == yhat:
                out_char = char
                break
        # Append the predicted character to the input text
        in_text += out_char
    return in_text

# Generate and print a sequence
print(generate_seq(model, mapping, 10, 'king', 10))

