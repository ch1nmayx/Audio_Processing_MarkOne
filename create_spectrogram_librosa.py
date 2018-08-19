"""
Author - Chinmay Sinha

Project - Music Genre Classification using Deep Learning

Description - Creation of a Spectrogram of an audio file using librosa library

"""

import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

def create_spectrogram_librosa(filename_w_format):
    """
    argument: the name of the file witht he format eg. 'coldplay.mp3'
    returns: the mel-spectrogram plot of the argument
    
    """
    
    y, sr = librosa.load(str(filename_w_format))
    
    # Returning an Array
    librosa.feature.melspectrogram(y=y, sr=sr)
    
    D = np.abs(librosa.stft(y))**2
    S = librosa.feature.melspectrogram(S=D)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                        fmax=8000)
    
    # Creating a mel-spectrogram plot using matplotlib.pyplot
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S,
                                                 ref=np.max),
                             y_axis='mel', fmax=8000,
                             x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('POP')
    plt.tight_layout()
    

create_spectrogram_librosa('pop.00000.au')