import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.fftpack import fft

total_time = 3  
sample_rate = 1024 
num_samples = 12 * sample_rate
t = np.linspace(0, total_time, num_samples)
N = 6

left_hand_notes = [130, 130, 146, 130, 174, 164] 
right_hand_notes = [261, 261, 293, 261, 349, 329]  
note_start_times = [0,0.5,1,1.5,2,2.5]
note_durations = [0.45,0.45,0.45,0.45,0.45,0.5] 

# Generate Song
signal = 0
for i in range(N):
    temp_signal = np.sin(2 * np.pi * left_hand_notes[i] * t) + np.sin(2 * np.pi * right_hand_notes[i] * t)
    x = np.where(np.logical_and(t >= note_start_times[i], t <= note_start_times[i] + note_durations[i]), temp_signal , 0)
    signal += x

# Plot and play original signal in time domain
plt.plot(t, signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Generated Song in time domain')
plt.show()

sd.play(signal, 3 * 1024)
sd.wait()

# Plot original signal in frequency domain
signal_in_freq_domain = fft(signal)
M = 3 * 1024
f = np.linspace(0 , 512, M // 2)
signal_in_freq_domain = 2 / M * np.abs(signal_in_freq_domain[0:M // 2])


plt.plot(f, signal_in_freq_domain)
plt.xlabel('Frequency (hz)')
plt.ylabel('Amplitude')
plt.title('Generated Song in frequency domain')
plt.show()


# Generating noise
f_noise_1, f_noise_2 = np.random.randint(0, 512, 2)
noise = np.sin(2 * np.pi * f_noise_1 * t) + np.sin(2 * np.pi * f_noise_2 * t)
x_noise = signal + noise

# Plot and play signal with noise in time domain
plt.plot(t, x_noise)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Generated Song with Noise in time domain')
plt.show()

sd.play(x_noise, 3 * 1024)
sd.wait()

# Plot signal with noise in frequency domain
signal_with_noise_in_freq_domain = fft(x_noise)
signal_with_noise_in_freq_domain = 2 / M * np.abs(signal_with_noise_in_freq_domain[0:M // 2])

plt.plot(f, signal_with_noise_in_freq_domain)
plt.xlabel('Frequency (hz)')
plt.ylabel('Amplitude')
plt.title('Generated Song with Noise in frequency domain')
plt.show()

# Find the two random frequencies
higher_freq_in_noisy = np.where(signal_with_noise_in_freq_domain > np.ceil(np.max(x)))
print(higher_freq_in_noisy)
index1 = higher_freq_in_noisy[0][0]
index2 = higher_freq_in_noisy[0][1]
fn1_new = int(f[index1])
fn2_new =  int(f[index2])

# Filter the noise
xfiltered = x_noise - (np.sin(2 * np.pi * fn1_new * t) + np.sin(2 * np.pi * fn2_new * t))

# Plot and play filtered signal in time domain
plt.plot(t, xfiltered)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Filtered Song in time domain')
plt.show()

sd.play(xfiltered, 3 * 1024)
sd.wait()

# Plot filtered signal in frequency domain
filtered_signal_in_freq_domain = fft(xfiltered)
filtered_signal_in_freq_domain = 2 / M * np.abs(filtered_signal_in_freq_domain[0:np.int(M/2)])

plt.plot(f, filtered_signal_in_freq_domain)
plt.xlabel('Frequency (hz)')
plt.ylabel('Amplitude')
plt.title('Filtered Song in frequency domain')
plt.show()
