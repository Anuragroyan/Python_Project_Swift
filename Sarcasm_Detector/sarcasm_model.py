import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
import numpy as np   
import json 
import coremltools as ct 

data = [
   {"headline":"Oh you fixed it? I barely even noticed it was broken in the first place","is_sarcastic": 4},
   {"headline":"Well that was certainly a productive meeting","is_sarcastic": 4},
   {"headline":"I'm so glad you shared that incredibly obvious observation","is_sarcastic": 3},
   {"headline":"You're right I totally forgot that was an option","is_sarcastic": 3},
   {"headline":"I'm just thrilled to be working late again", "is_sarcastic": 5},
   {"headline":"That's a fantastic idea why didn't anyone else think of it?","is_sarcastic": 4},
   {"headline":"I'm sure your terrible driving is just a unique personal style","is_sarcastic": 2},
   {"headline":"My day just wouldn't be complete without this minor inconvenience","is_sarcastic": 5},
   {"headline":"Wow you're really good at pointing out things I already know","is_sarcastic": 3},
   {"headline":"I had no idea you were such a genius Please enlighten me further","is_sarcastic": 5},
   {"headline":"Thanks for the profound insight it really cleared things up","is_sarcastic": 4},
   {"headline":"I'm perfectly fine just bleeding internally","is_sarcastic": 5},
   {"headline":"Oh is that a new concept? I've never heard of it before","is_sarcastic": 4},
   {"headline":"I'm always eager to hear about your latest accomplishments","is_sarcastic": 3},
   {"headline":"That's exactly what I needed right now","is_sarcastic": 5},
   {"headline":"I'm so glad we could have this meaningful conversation","is_sarcastic": 4},
   {"headline":"Of course because things always go according to plan","is_sarcastic": 4},
   {"headline":"I'm just overflowing with excitement","is_sarcastic": 5},
   {"headline":"What a brilliant solution to a problem that didn't exist","is_sarcastic": 5},
   {"headline":"I must have misheard you; I thought you said something intelligent","is_sarcastic": 5},
   {"headline":"Please don't rush on my account","is_sarcastic": 3},
   {"headline":"You're truly a master of understatement","is_sarcastic": 4},
   {"headline":"I can practically feel the enthusiasm radiating from you","is_sarcastic": 4},
   {"headline":"I appreciate your valuable contribution to the discussion","is_sarcastic": 3},
   {"headline":"Clearly you've put a lot of thought into that","is_sarcastic": 4},
   {"headline":"I'm sure that will be incredibly helpful","is_sarcastic": 3},
   {"headline":"Well isn't that just special?","is_sarcastic": 4},
   {"headline":"I had no idea you were capable of such feats of logic","is_sarcastic": 2},
   {"headline":"It's truly a pleasure to be graced by your presence","is_sarcastic": 4},
   {"headline":"I'm so relieved you're here to explain the obvious to me","is_sarcastic": 3},
   {"headline":"I'm feeling very bad nowsdays","is_sarcastic": 0},
   {"headline": "the weather is absolutely amazing (it's raining cats and dogs)", "is_sarcastic": 1},
   {"headline": "local man wins lottery", "is_sarcastic": 0},
   {"headline": "new study shows coffee is good for you", "is_sarcastic": 0},
   {"headline": "sure, because skipping sleep makes you smarter", "is_sarcastic": 1}
]

sentences = [item["headline"] for item in data]
labels = [item["is_sarcastic"] for item in data]

# tokenization
vocab_size = 1000
embedding_dim = 32
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=max_length,padding=padding_type,truncating=trunc_type)

# model
model = Sequential([
    Embedding(vocab_size,embedding_dim,input_length=max_length),
    GlobalAveragePooling1D(),
    Dense(24, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

# train model
model.fit(padded, np.array(labels), epochs=30)

# save tokenizer
import pickle
with open('tokenizer.pk1', 'wb') as f:
    pickle.dump(tokenizer, f)

# convert to core ml
class SarcasmDetector(tf.Module):
    def __init__(self,model):
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec([None, max_length], tf.int32)])
    def __call__(self, x):
        return {"prediction": self.model(x)}

# convert the tensorflow model
detector = SarcasmDetector(model)
mlmodel = ct.convert(detector.model, source='tensorflow')
mlmodel.save("SarcasmDetector.mlpackage")
# mlmodel.save("SarcasmDetector.mlmodel")        


