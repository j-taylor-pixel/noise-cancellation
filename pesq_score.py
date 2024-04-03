from scipy.io import wavfile
from pesq import pesq
from scipy.io import wavfile
from scipy.signal import resample
import os


def resample_file(filename='test_rec.wav'):
    # Read the original file
    orig_rate, orig_signal = wavfile.read(filename)

    # Calculate the number of samples in the resampled signal
    new_rate = 16000  # New sampling rate
    num_samples = int(len(orig_signal) * new_rate / orig_rate)

    # Resample the signal
    resampled_signal = resample(orig_signal, num_samples)

    # Write the resampled signal to a new file
    wavfile.write(f'resampled_{filename}', new_rate, resampled_signal.astype(orig_signal.dtype))

    return f'resampled_{filename}'


def get_pesq_score(file_one='resampled_test_rec.wav', file_two='resampled_sensor_1.wav'):
    # Read the reference and test files
    ref_rate, ref_signal = wavfile.read(file_one)
    deg_rate, deg_signal = wavfile.read(file_two)

    # Ensure the sample rates match
    assert ref_rate == deg_rate

    # Compute the PESQ score
    score = pesq(ref_rate, ref_signal, deg_signal, 'wb')  # 'wb' for wide-band mode

    print(f'PESQ score: {score}')
    if os.path.exists(file_one):
        os.remove(file_one)
    if os.path.exists(file_two):
        os.remove(file_two)
    
    return score


# trim audio file to only 30 seconds
def trim_audio(filename):
    # Read the original file
    rate, signal = wavfile.read(filename)

    # Calculate the number of samples corresponding to 30 seconds
    num_samples = 30 * rate

    # Trim the signal
    trimmed_signal = signal[:num_samples]

    # Write the trimmed signal to a new file
    wavfile.write(f'trimmed_{filename}', rate, trimmed_signal)
    if os.path.exists(filename):
        os.remove(filename)
    
    

    return f'trimmed_{filename}'

def pesq_of_two_raw(file_one, file_two):
    resampled_one = resample_file(file_one)
    resampled_two = resample_file(file_two)
    trim_one = trim_audio(resampled_one)
    trim_two = trim_audio(resampled_two)


    return get_pesq_score(trim_one, trim_two)

pesq_of_two_raw('test_rec.wav', 'uni_mix_reduced_output.wav')
# 1.26, 1.13, 1.22 for sensor 1, 2, 3 compared to test_rec
# unix mixed wave was 1.03
# proccessed2 was 1.04