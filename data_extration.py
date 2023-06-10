import os
from math import log

from scipy.io import wavfile

from analysis import note_to_harmonics
from constants import INSTRUMENTS

data = []

for instrument_id, instrument in enumerate(INSTRUMENTS):
    print('Processing ' + instrument + '...')
    for filename in os.listdir('samples/' + instrument):
        try:
            fs, wav_data = wavfile.read('samples/' + instrument + '/' + filename)
        except ValueError:
            continue
        print(filename)
        for i in range(int(len(wav_data) / 1024)):
            chunk = wav_data[i * 1024:(i + 1) * 1024]
            # skip chunk if it is silent
            if max(chunk) < 1000:
                continue
            harmonics = note_to_harmonics(fs, chunk, 20)
            harmonics = [harmonics[0]] + [log(x) if x > 0 else -float('inf') for x in harmonics[1:]]
            shift = 1 - harmonics[1]
            for i in range(1, len(harmonics)):
                harmonics[i] += shift
            print(harmonics)
            answer = [0] * len(INSTRUMENTS)
            answer[instrument_id] = 1
            data.append([harmonics, answer])


print('Data size: ' + str(len(data)))

# save data to file data.data
with open('log_data.data', 'w') as f:
    for i in range(len(data)):
        f.write(" ".join(map(str, data[i][0])) + " " + " ".join(map(str, data[i][1])) + '\n')
