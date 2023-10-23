import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, Conv1D, MaxPooling1D, Bidirectional, 
                                     LSTM, Dense, Dropout, BatchNormalization)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import Add
from tensorflow.keras.optimizers.schedules import ExponentialDecay



# Adjusted path to the CSV file
data_path = "/home/24694266/DataScience344/Project/RNNModels/Filterd.csv"
data = pd.read_csv(data_path)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    print("GPU devices available:")
    for device in gpu_devices:
        device_name = device.name
        compute_capability = tf.test.gpu_device_name()  # Get compute capability
        print(f"Name: {device_name}, Compute Capability: {compute_capability}")
    tf.config.experimental.set_visible_devices(gpu_devices[0], 'GPU')  # Use the first GPU
else:
    print("No GPU devices available. TensorFlow will run on CPU.")

# 1. Randomly sample 100,000 observations from your data
data_sample = data.sample(n=50000, random_state=42)

# 2. Apply the preprocessing steps to this subset
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(data_sample['Lyrics_Processed'])

word_index = tokenizer.word_index
max_words = len(word_index) + 1

sequences = tokenizer.texts_to_sequences(data_sample['Lyrics_Processed'])
padded_sequences = pad_sequences(sequences, maxlen=100, truncating='post', padding='post')

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(data_sample['genre'])
labels_one_hot = to_categorical(labels_encoded, num_classes=6)
labels = labels_one_hot

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

maxlen = 100
embedding_dim = 16

with tf.device('/GPU:0'):
    regularization_strength = 0.02
    
    model = Sequential()
    
    # Embedding layer
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=maxlen))
    
    # First set of convolutional + pooling layers
    model.add(Conv1D(16, kernel_size=2, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    
    # Second set of convolutional + pooling layers
    model.add(Conv1D(16, kernel_size=2, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    
    # Bidirectional LSTM layers
    model.add(Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer=l2(regularization_strength))))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(32, kernel_regularizer=l2(regularization_strength))))
    
    
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(regularization_strength)))
    model.add(Dropout(0.5)) 
    model.add(Dense(64, activation='relu', kernel_initializer=GlorotUniform(), kernel_regularizer=l2(regularization_strength)))
    model.add(Dropout(0.5))  # Optional: Adding dropout after the dense layer can be beneficial for preventing overfitting
    
    model.add(Dense(6, activation='softmax'))
    
    optimizer = SGD(learning_rate=0.01, momentum=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, restore_best_weights=True)
    model_save_path = "/home/24694266/DataScience344/Project/RNNModels/Model1/model_epoch_{epoch:02d}.h5"
    model_checkpoint = ModelCheckpoint(model_save_path, save_best_only=False, verbose=1)
    csv_logger = CSVLogger('training_log.csv', append=True)
    callbacks_list = [model_checkpoint, csv_logger]

    # Train the model
    history = model.fit(X_train, y_train, epochs=500, batch_size=64, validation_data=(X_test, y_test),verbose=1, callbacks=callbacks_list)
