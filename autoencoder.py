import glob
import pandas
import sys
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from keras.datasets import mnist
from keras.regularizers import l1
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

LANDMARKS_MASKED_PATH = 'processed/landmarks/masked'
LANDMARKS_UNMASKED_PATH = 'processed/landmarks/unmasked'
AUDIO_PATH = 'processed/audio'


def read_features():
    labels = []
    features = []
    size = 2184
    for masked, unmasked, audio in zip(glob.iglob(LANDMARKS_MASKED_PATH + '/**/*.csv', recursive=True),
                                       glob.iglob(LANDMARKS_UNMASKED_PATH + '/**/*.csv', recursive=True),
                                       glob.iglob(AUDIO_PATH + '/**/*.csv', recursive=True)):
        try:
            audio_data = np.genfromtxt(audio, delimiter=',').flatten()
            landmark_data = np.genfromtxt(masked, delimiter=',').flatten()
            feature = np.concatenate([landmark_data, audio_data], axis=0).flatten()
            if feature.shape[0] != size:
                continue

            features.append(feature)
            labels.append(np.genfromtxt(unmasked, delimiter=',').flatten())
        except Exception:
            pass
            # print(sys.exc_info()[0])
    return np.array(features), np.array(labels)


if __name__ == '__main__':
    features, labels = read_features()
    print('Labels: ', labels.shape)
    print('Features: ', features.shape)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.15)

    input_size = features.shape[1]
    hidden_size = input_size//2
    code_size = input_size//4
    output_size = labels.shape[1]
    hidden_size_2 = input_size//6

    input_features = Input(shape=(input_size,))
    hidden_1 = Dense(hidden_size, activation='relu')(input_features)
    code = Dense(code_size, activation='relu')(hidden_1)
    hidden_2 = Dense(hidden_size_2, activation='relu')(code)
    output_landmarks = Dense(output_size, activation='relu')(hidden_2)

    autoencoder = Model(input_features, output_landmarks)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_train, y_train, epochs=20, batch_size=1, validation_split=0.1275)

    scores = autoencoder.evaluate(X_test, y_test, batch_size=1)
