import noisereduce as nr
import soundfile as sf
import matplotlib.pyplot as plt

data1, samplerate1 = sf.read('sensor_1.wav')
data2, samplerate2 = sf.read('sensor_2.wav')
data3, samplerate3 = sf.read('sensor_3.wav')

# Mix audio. Scale isn't really important; it just changes the volume
data = (data1 + data2 + data3)

reduced_noise = nr.reduce_noise(y = data, sr=samplerate1, n_std_thresh_stationary=2.0,stationary=False)
sf.write("processed2.wav", reduced_noise, samplerate1)
# Plot
plt.plot(data, label="Input")
plt.plot(reduced_noise, label="Output")
plt.legend()
plt.title("Non-stationary Noise Reduction")
plt.savefig('Non-stationary Noise Reduction')
