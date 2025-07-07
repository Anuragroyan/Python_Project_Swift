import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import coremltools as ct

# 1. Dummy data
data = {
    'label': [
        'ham', 'spam', 'ham', 'spam', 'ham',
        'spam', 'ham', 'spam', 'ham', 'spam',
        'ham', 'spam', 'ham', 'spam', 'ham',
        'spam', 'ham', 'spam', 'ham', 'spam',
        'ham', 'spam', 'ham', 'spam', 'ham',
        'spam', 'ham', 'spam', 'ham', 'spam',
        'ham', 'spam', 'ham', 'spam', 'ham',
        'spam', 'ham', 'spam', 'ham', 'spam',
        'ham', 'spam', 'ham', 'spam', 'ham'
    ],
    'message': [
        'Hey, how are you doing?',
        'Win a FREE iPhone now!!!',
        'Are we still on for lunch?',
        'Congratulations, you have won a prize!',
        'Let’s catch up tomorrow.',
        'Claim your $1000 gift card now!',
        'Can you send me the notes?',
        'Limited time offer, act now!',
        'I\'ll be there in 10 minutes.',
        'Exclusive deal just for you!',
        'Happy birthday!',
        'Get cash fast, no credit check!',
        'What time is the meeting?',
        'You have been selected for a survey.',
        'Dinner at 7?',
        'Earn money from home easily!',
        'Did you complete the assignment?',
        'Urgent! Your account is compromised.',
        'Thanks for the update!',
        'Click here to win a trip!',
        'Let\'s meet at the coffee shop.',
        'You are pre-approved for a loan!',
        'Please find the attachment.',
        'Lowest insurance rates guaranteed!',
        'Join the Zoom call now.',
        'Congratulations! You’ve won!',
        'Where should we go today?',
        'Unlock your special bonus now!',
        'See you at the party!',
        'Important notice: Final warning!',
        'Call me when you\'re free.',
        'You’re a lucky winner!',
        'How’s the new job?',
        'Get rich quick with this scheme!',
        'Let’s plan a movie night.',
        'Your account will be deactivated!',
        'Lunch at our usual place?',
        'Hurry up! Limited slots left!',
        'Good luck with your exam!',
        'Don\'t miss this limited-time deal!',
        'That sounds perfect!',
        'Act now to claim your reward!',
        'Catch you later!',
        'Earn $5000 from your phone!',
        'Meeting postponed to tomorrow.'  # ← added 45th message
    ]
}


# 2. Preprocess
texts = data['message']
labels = [1 if label == 'spam' else 0 for label in data['label']]  # 1=spam, 0=ham

tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, padding='post', maxlen=20)

X = np.array(padded)
y = np.array(labels)

# 3. Build model
model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=20),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. Train model
model.fit(X, y, epochs=10, verbose=1)

# 5. Convert to CoreML
input_shape = (1, 100)  # or whatever maxlen you used
mlmodel = ct.convert(model, inputs=[ct.TensorType(name="embedding_input", shape=input_shape)])
# mlmodel = mlmodel = ct.convert(model, inputs=[ct.TensorType(name="embedding_input", shape=input_shape)])


# 6. Save CoreML model
mlmodel.save("SpamClassifier.mlpackage")

# 7. Save tokenizer for iOS pre-processing use
import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

mlmodel = ct.models.MLModel("SpamClassifier.mlpackage")
print(mlmodel.output_description)
# print(mlmodel.output_description["SpamClassifier.mlpackage"])
print(mlmodel.output_description)
print("Available outputs:", mlmodel.output_description._fd_spec.description.output)
