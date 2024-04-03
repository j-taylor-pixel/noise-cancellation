import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft



def plot_waveform(filename='test_rec.wav'):
    wav_file = wave.open(filename, 'r')

    # Get the frame rate
    framerate = wav_file.getframerate()

    # Calculate the number of frames that correspond to 30 seconds
    num_frames = 30 * framerate

    # Read the corresponding frames from the file
    signal = wav_file.readframes(num_frames)
    signal = np.frombuffer(signal, dtype='int16')

    # Time axis in seconds
    time = np.linspace(0., len(signal) / framerate, num=len(signal))

    # Close the file
    wav_file.close()

    # Create a new figure
    plt.figure()

    # Plot the signal
    plt.plot(time, signal)

    # Label the axes
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    # Here you can specify the title
    plt.title(f'Waveform of {filename}.wav')

    # Display the plot
    plt.savefig(f'{filename}.png')
    return

# create histogram of amplitdues of audio file
def histogram_of_amplitudes(filename):
    wav_file = wave.open(filename, 'r')

    # Get the frame rate
    framerate = wav_file.getframerate()

    # Calculate the number of frames that correspond to 30 seconds
    num_frames = 30 * framerate

    # Read the corresponding frames from the file
    signal = wav_file.readframes(num_frames)
    signal = np.frombuffer(signal, dtype='int16')

    # Time axis in seconds
    time = np.linspace(0., len(signal) / framerate, num=len(signal))

    # Close the file
    wav_file.close()

    # Create a new figure
    plt.figure()

    # Plot the histogram
    plt.hist(signal, bins=100)

    # print highest and lowest amplitude of signal
    print(f'Highest amplitude: {max(signal)}, lowest amplitude: {min(signal)}')

    # Label the axes
    plt.xlabel('Amplitude')
    plt.ylabel('Count')

    # Here you can specify the title
    plt.title(f'Histogram of amplitudes of {filename}.wav')

    # Display the plot
    plt.savefig(f'histograms/amp_hist_{filename}.png')
    return max(signal) - min(signal)


def histogram_of_frequencies(filename):
    # Open the file in read mode
    wav_file = wave.open(filename, 'r')

    # Read frames from the file
    signal = wav_file.readframes(-1)
    signal = np.frombuffer(signal, dtype='int16')

    # Get the frame rate
    framerate = wav_file.getframerate()

    # Close the file
    wav_file.close()

    # Compute the FFT of the signal and take the absolute value to get the magnitude
    spectrum = np.abs(fft(signal))

    # Compute the frequencies corresponding to the spectrum and only consider the positive frequencies
    freq = np.fft.fftfreq(len(spectrum), 1 / framerate)
    mask = freq > 0
    spectrum = spectrum[mask]
    freq = freq[mask]

    # Create a new figure
    plt.figure()

    # Plot the histogram of the frequencies
    plt.hist(freq, bins=100, weights=spectrum)

    # Label the axes
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')

    # Here you can specify the title
    plt.title(f'Histogram of frequencies of {filename}')

    # Save the plot
    plt.savefig(f'histograms/freq_hist_{filename}.png')
    return


def histogram_of_volume_db(filename):
    # Open the file in read mode
    wav_file = wave.open(filename, 'r')
    # Read frames from the file
    signal = wav_file.readframes(-1)
    signal = np.frombuffer(signal, dtype='int16')
    # Close the file
    wav_file.close()
    # Convert the signal to volume in dB
    volume_db = 20 * np.log10(np.abs(signal))
    # Remove -inf values
    volume_db = volume_db[volume_db != -np.inf]

    # Create a new figure
    plt.figure()
    # Plot the histogram of the volume in dB
    plt.hist(volume_db, bins='auto')
    # Label the axes
    plt.xlabel('Volume [dB]')
    plt.ylabel('Count')
    # Here you can specify the title
    plt.title(f'Histogram of volume (dB) of {filename}')
    # Save the plot
    plt.savefig(f'histograms/vol_db_hist{filename}.png')
    return

def generate_histograms(filename):
    #plot_waveform(filename)
    histogram_of_amplitudes(filename)
    histogram_of_frequencies(filename)
    histogram_of_volume_db(filename)
    return

generate_histograms('test_rec.wav') # test set from pixel
generate_histograms('processed2.wav') # simple addition fusion
generate_histograms('sensor_1.wav') # raw and normalized
generate_histograms('sensor_2.wav') # raw and normalized
generate_histograms('sensor_3.wav') # raw and normalized
generate_histograms('uni_mix_reduced_output.wav') # 
