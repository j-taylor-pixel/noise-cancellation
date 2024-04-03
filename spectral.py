import librosa
import numpy as np

# Load the audio files
audio_file1 = 'data/sensor_1.wav'
audio_file2 = 'data/sensor_2.wav'
threshold1 = 0.01
threshold2 = 0.05
y1, sr1 = librosa.load(audio_file1, sr=None)
y2, sr2 = librosa.load(audio_file2, sr=None)

# Compute the short-time Fourier transform (STFT) for both audio signals
D1 = np.abs(librosa.stft(y1))
D2 = np.abs(librosa.stft(y2))

# Compute the mean magnitude spectrum for both signals
mean_D1 = np.mean(D1, axis=1)
mean_D2 = np.mean(D2, axis=1)

# Compute a mask to remove quiet sounds that both signals share
mask = np.logical_or(mean_D1 > threshold1, mean_D2 > threshold2)

# Apply the mask to the magnitude spectra of both signals
D1_filtered = D1 * mask[:, np.newaxis]
D2_filtered = D2 * mask[:, np.newaxis]

# Inverse STFT to obtain the filtered signals
y1_filtered = librosa.istft(D1_filtered)
y2_filtered = librosa.istft(D2_filtered)

# Save the filtered audio files
librosa.output.write_wav('filtered_audio_file1.wav', y1_filtered, sr1)
librosa.output.write_wav('filtered_audio_file2.wav', y2_filtered, sr2)
