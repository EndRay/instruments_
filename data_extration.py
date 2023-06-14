import os
import random
from math import log

import numpy as np
from scipy.io import wavfile

from analysis import note_to_harmonics
from constants import INSTRUMENTS, CHUNK_SIZE, SILENT_CHUNK_THRESHOLD


DATA_AMOUNTS = [
    100,
    3000, 3000, 3000,
    3000, 3000, 3000,
    3000, 3000, 3000,
    3000, 2000, 2000,
    1000, 100
]
MAX_INSTRUMENTS = 10
NO_SILENT_CHUNKS = True
MAX_CHUNKS_SHIFT = 20

data = [[] for _ in range(MAX_INSTRUMENTS+1)]

INSTRUMENTS_FILES = {instrument: [] for instrument in INSTRUMENTS}

for instrument_id, instrument in enumerate(INSTRUMENTS):
    for filename in os.listdir('samples/' + instrument):
        # skip quiet samples
        if '-pp-' in filename:
            continue
        try:
            wavfile.read('samples/' + instrument + '/' + filename)
            INSTRUMENTS_FILES[instrument].append('samples/' + instrument + '/' + filename)
        except ValueError:
            pass


while any(len(data[i]) < DATA_AMOUNTS[i] for i in range(len(data))):
    print('+'.join(map(str, [len(data[i]) for i in range(len(data))])))
    # instruments_amount = random.randint(1, MAX_INSTRUMENTS)
    instruments_amount = 14
    instruments = random.sample(INSTRUMENTS, instruments_amount)
    samples = []
    for instrument in instruments:
        samples.append(random.choice(INSTRUMENTS_FILES[instrument]))
    samples = [wavfile.read(sample) for sample in samples]
    longest_sample_len = max([len(sample[1]) for sample in samples])
    shifts = [0] + [random.randint(0, MAX_CHUNKS_SHIFT) for _ in range(instruments_amount-1)]
    for chunk_id in range(longest_sample_len // CHUNK_SIZE):
        result = [0] * len(INSTRUMENTS)
        chunk_sum = np.zeros(CHUNK_SIZE)
        for instrument_id, instrument in enumerate(instruments):
            chunk = samples[instrument_id][1][(shifts[instrument_id] + chunk_id) * CHUNK_SIZE:(shifts[instrument_id] + chunk_id + 1) * CHUNK_SIZE]
            if len(chunk) < CHUNK_SIZE:
                continue
            if max(chunk) / 32768 > SILENT_CHUNK_THRESHOLD:
                result[INSTRUMENTS.index(instrument)] = 1
            chunk_sum += chunk
        cnt = sum(result)
        if cnt > MAX_INSTRUMENTS or len(data[cnt]) >= DATA_AMOUNTS[cnt]:
            continue
        chunk_sum /= MAX_INSTRUMENTS
        chunk_sum /= 32768
        data[cnt].append([chunk_sum, result])


print('Data size: ' + str(sum(len(data[i]) for i in range(MAX_INSTRUMENTS))))

# save data to file data.data
with open('shifted_multiple_balanced.data', 'w') as f:
    for cnt in range(len(data)):
        for i in range(len(data[cnt])):
            f.write(" ".join(map(str, data[cnt][i][0])) + " " + " ".join(map(str, data[cnt][i][1])) + '\n')
