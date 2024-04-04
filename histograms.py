import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

def get_signal(filename):
    wav_file = wave.open(filename, 'r')
    signal = wav_file.readframes(-1)
    signal = np.frombuffer(signal, dtype='int16')
    wav_file.close()
    return signal

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
    plt.hist(volume_db, bins=10)
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

def number_of_samples(filename):
    wav_file = wave.open(filename, 'r')
    framerate = wav_file.getframerate()
    num_frames = wav_file.getnframes()
    wav_file.close()
    return num_frames

def count_samples():
    for file in ['test_rec.wav', 'processed2.wav', 'sensor_1.wav', 'sensor_2.wav', 'sensor_3.wav', 'uni_mix_reduced_output.wav']:
        print(f'Number of samples in {file}: {number_of_samples(file)}')
        # all samples have 9595771 frames
    return

def manhattan_distance_of_volume(file_one, file_two):
    wav_file_one = wave.open(file_one, 'r')
    wav_file_two = wave.open(file_two, 'r')
    signal_one = wav_file_one.readframes(-1)
    signal_two = wav_file_two.readframes(-1)
    signal_one = np.frombuffer(signal_one, dtype='int16')
    signal_two = np.frombuffer(signal_two, dtype='int16')
    wav_file_one.close()
    wav_file_two.close()
    if len(signal_two) > len(signal_one):
        signal_two = signal_two[::2]
    elif len(signal_one) > len(signal_two):
        signal_one = signal_one[::2]

    volume_db_one = 20 * np.log10(np.abs(signal_one)+1) # warning when log 0
    volume_db_two = 20 * np.log10(np.abs(signal_two)+1)
    #volume_db_one = volume_db_one[volume_db_one != -np.inf]
    #volume_db_two = volume_db_two[volume_db_two != -np.inf]
    manhattan_distance = np.sum(np.abs(volume_db_one - volume_db_two))
    print(f'Manhattan distance of volume between {file_one} and {file_two}: {manhattan_distance}')
    return manhattan_distance

def manhattan_distance_amplitude(file_one, file_two):
    wav_file_one = wave.open(file_one, 'r')
    wav_file_two = wave.open(file_two, 'r')
    signal_one = wav_file_one.readframes(-1)
    signal_two = wav_file_two.readframes(-1)
    signal_one = np.frombuffer(signal_one, dtype='int16')
    signal_two = np.frombuffer(signal_two, dtype='int16')

    # remove every second value from signal_two if its longer than signal_one
    if len(signal_one) > len(signal_two):
        signal_one = signal_one[::2]

    wav_file_one.close()
    wav_file_two.close()
    manhattan_distance = np.sum(np.abs(signal_one - signal_two))
    manhattan_distance = "{:.3e}".format(manhattan_distance)
    print(f'Manhattan distance of amplitude between {file_one} and {file_two}: {manhattan_distance}')
    return manhattan_distance

def manhattan_distance_frequence(file_one, file_two):
    wav_file_one = wave.open(file_one, 'r')
    wav_file_two = wave.open(file_two, 'r')
    signal_one = wav_file_one.readframes(-1)
    signal_two = wav_file_two.readframes(-1)
    signal_one = np.frombuffer(signal_one, dtype='int16')
    signal_two = np.frombuffer(signal_two, dtype='int16')

    # remove every second value from signal_two if its longer than signal_one
    if len(signal_one) > len(signal_two):
        signal_one = signal_one[::2]

    wav_file_one.close()
    wav_file_two.close()
    spectrum_one = np.abs(fft(signal_one))
    spectrum_two = np.abs(fft(signal_two))
    manhattan_distance = np.sum(np.abs(spectrum_one - spectrum_two))
    # format manhattan_distance as scientific notation, with 3 decimal places
    manhattan_distance = "{:.3e}".format(manhattan_distance)
    print(f'Manhattan distance of {manhattan_distance} between frequencies of {file_one} and {file_two}: ')
    return manhattan_distance

def normalize_amplitudes(file_one, file_two):    
    wav_file_one = wave.open(file_one, 'r')
    wav_file_two = wave.open(file_two, 'r')
    signal_one = wav_file_one.readframes(-1)
    signal_two = wav_file_two.readframes(-1)
    signal_one = np.frombuffer(signal_one, dtype='int16')
    signal_two = np.frombuffer(signal_two, dtype='int16')
    wav_file_one.close()
    wav_file_two.close()
    # normalize signal_two to signal_one
    signal_two = signal_two * np.average(signal_one) / np.average(signal_two)
    print(f'file 1: {np.average(signal_one)}, file 2: {np.average(signal_two)}')
    #save signal_two to new file
    # Convert the normalized signal back to bytes
    signal_two_bytes = signal_two.astype(np.int16).tobytes()

    # Create a new wave file
    with wave.open('volume_segment_pcm_norm.wav', 'w') as wav_file:
        # Use the same parameters as the original file
        wav_file.setparams(wav_file_one.getparams())
        # Write the normalized signal to the new file
        wav_file.writeframes(signal_two_bytes)
    return signal_one, signal_two

def compare_amp_hist(file_one, file_two):
    wav_file_one = wave.open(file_one, 'r')
    wav_file_two = wave.open(file_two, 'r')

    # Get the frame rate
    framerate = wav_file_one.getframerate()
    # Calculate the number of frames that correspond to 30 seconds
    num_frames = 30 * framerate

    # Read the corresponding frames from the file
    signal_one = wav_file_one.readframes(num_frames)
    signal_one = np.frombuffer(signal_one, dtype='int16')
    signal_two = wav_file_two.readframes(num_frames)
    signal_two = np.frombuffer(signal_two, dtype='int16')
    min_of_both = min(min(signal_one), min(signal_two))
    max_of_both = max(max(signal_one), max(signal_two))
    hist_1 = np.histogram(a=signal_one, bins=10, range=(min_of_both, max_of_both))
    hist_2 = np.histogram(a=signal_two, bins=10, range=(min_of_both, max_of_both))

    # close the files
    wav_file_one.close()
    wav_file_two.close()
    # compute difference between histograms
    hist_diff = sum(np.abs(hist_1[0] - hist_2[0]))
    hist_diff = "{:.3e}".format(hist_diff)
    print(f'amplitude hist diff: {hist_diff} between {file_one} and {file_two}')
    return hist_diff

def compare_freq_hist(file_one, file_two):
    # Open the file in read mode
    wav_file_one = wave.open(file_one, 'r')
    wav_file_two = wave.open(file_two, 'r')
    # Read frames from the file
    signal_one = wav_file_one.readframes(-1)
    signal_one = np.frombuffer(signal_one, dtype='int16')
    signal_two = wav_file_two.readframes(-1)
    signal_two = np.frombuffer(signal_two, dtype='int16')
    # Get the frame rate
    framerate = wav_file_one.getframerate()

    # Close the file
    wav_file_one.close()
    wav_file_two.close()
    # Compute the FFT of the signal and take the absolute value to get the magnitude
    spectrum_one = np.abs(fft(signal_one))
    spectrum_two = np.abs(fft(signal_two))

    # Compute the frequencies corresponding to the spectrum and only consider the positive frequencies
    freq_one = np.fft.fftfreq(len(spectrum_one), 1 / framerate)
    freq_two = np.fft.fftfreq(len(spectrum_two), 1 / framerate)

    mask_one = freq_one > 0
    mask_two = freq_two > 0
    spectrum_one = spectrum_one[mask_one]
    spectrum_two = spectrum_two[mask_two]
    freq_one = freq_one[mask_one]
    freq_two = freq_two[mask_two]

    min_both = min(min(freq_one), min(freq_two))
    max_both = max(max(freq_one), max(freq_two))
    hist_1 = np.histogram(a=freq_one, bins=10, range=(min_both, max_both), weights=spectrum_one)
    hist_2 = np.histogram(a=freq_two, bins=10, range=(min_both, max_both), weights=spectrum_two)

    hist_diff = sum(np.abs(hist_1[0] - hist_2[0]))
    # format hist_diff as scientific notation, with 3 decimal places
    hist_diff = "{:.3e}".format(hist_diff)
    print(f'frequency hist diff: {hist_diff} between {file_one} and {file_two}')
    return hist_diff



if False:
    generate_histograms('test_rec.wav') # test set from pixel
    generate_histograms('processed2.wav') # simple addition fusion
    generate_histograms('sensor_1.wav') # raw and normalized
    generate_histograms('sensor_2.wav') # raw and normalized
    generate_histograms('sensor_3.wav') # raw and normalized
    generate_histograms('uni_mix_reduced_output.wav') # 
if False:
    generate_histograms('volume_segment_pcm_norm.wav')

if False:
    for file in ['test_rec.wav', 'sensor_1.wav', 'sensor_2.wav', 'sensor_3.wav', 'processed2.wav', 'uni_mix_reduced_output.wav']:
        #manhattan_distance_frequence('test_rec.wav', file)
        manhattan_distance_amplitude('test_rec.wav', file)
if False: 
    number_of_samples('volume_segment_pcm_norm.wav')
    manhattan_distance_frequence('test_rec.wav', 'volume_segment_pcm_norm.wav')
    manhattan_distance_amplitude('test_rec.wav', 'volume_segment_pcm_norm.wav')

if False:
    normalize_amplitudes('test_rec.wav', 'volume_segment_pcm.wav')

if True:
    for file in ['sensor_1.wav', 'sensor_2.wav', 'sensor_3.wav', 'processed2.wav', 'uni_mix_reduced_output.wav', 'volume_segment_pcm_norm.wav']:
        compare_amp_hist('test_rec.wav', file)
        #compare_freq_hist('test_rec.wav', file)
