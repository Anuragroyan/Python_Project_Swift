import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import coremltools as ct

# Sample data
data = {
    "text": [
        "OMG I love this! #fun", 
        "Check out my new photo album!", 
        "Follow me on my journey!", 
        "Just retweeted this cool article", 
        "Had a great time with friends today", 
        "New post on my story!"
    ],
    "label": ["Twitter", "Facebook", "Instagram", "Twitter", "Facebook", "Instagram"]
}
df = pd.DataFrame(data)

# Label encoding
label_to_index = {label: idx for idx, label in enumerate(sorted(df["label"].unique()))}
index_to_label = {idx: label for label, idx in label_to_index.items()}
df["label_idx"] = df["label"].map(label_to_index)

# Tokenization
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.texts_to_sequences(df["text"])
padded_sequences = pad_sequences(sequences, maxlen=20)

# Neural network model
model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=20),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(len(label_to_index), activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded_sequences, df["label_idx"], epochs=30, verbose=0)

# Convert to CoreML
mlmodel = ct.convert(
    model,
    inputs=[ct.TensorType(name="embedding_input", shape=(1, 20), dtype=np.int32)],
    classifier_config=ct.ClassifierConfig(class_labels=[index_to_label[i] for i in range(len(index_to_label))])
)

# Save CoreML model
mlmodel.save("SocialMediaNNClassifier.mlmodel")

# Save tokenizer (optional, for preprocessing on iOS)
import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
