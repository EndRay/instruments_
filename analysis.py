import numpy as np

HARMONICS_TO_ANALYZE = 100


# wav of played note -> [frequency of note, amplitude of first harmonic, amplitude of second harmonic, ...]
def note_to_harmonics(fs, data, harmonics_count):
    fft_data = np.fft.ifft(data)
    fft_data = np.abs(fft_data)
    fft_data = fft_data[:int(len(fft_data) / 2)]

    peaks = []

    for i in range(1, len(fft_data) - 1):
        if fft_data[i - 1] < fft_data[i] and fft_data[i] > fft_data[i + 1]:
            freq = i * fs / len(fft_data)
            peaks.append((freq, fft_data[i]))

    peaks.sort(key=lambda x: x[1], reverse=True)

    # we assume that between three strongest harmonics there is a note
    note_frequency = 44100
    for peak in peaks[:3]:
        note_frequency = min(note_frequency, peak[0])

    answer = [note_frequency] + [0] * harmonics_count

    for peak in peaks[:HARMONICS_TO_ANALYZE]:
        harmonic = int(round(peak[0] / note_frequency))
        if harmonic <= 0 or harmonic > harmonics_count:
            continue
        answer[harmonic] = max(answer[harmonic], peak[1])

    return answer

