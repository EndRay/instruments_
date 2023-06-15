import os
import random
from math import log

import numpy as np
from scipy.io import wavfile

from analysis import note_to_harmonics
from constants import INSTRUMENTS, CHUNK_SIZE, SILENT_CHUNK_THRESHOLD

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


def extract_data(data_filename, data_amounts, max_chunks_shift=0, data_type='sample'):
    max_instruments = len(data_amounts) - 1
    data = [[] for _ in range(max_instruments + 1)]

    while any(len(data[i]) < data_amounts[i] for i in range(len(data))):
        print('+'.join(map(str, [len(data[i]) for i in range(len(data))])))
        # instruments_amount = random.randint(1, MAX_INSTRUMENTS)
        instruments_amount = 14
        instruments = random.sample(INSTRUMENTS, instruments_amount)
        samples = []
        for instrument in instruments:
            samples.append(random.choice(INSTRUMENTS_FILES[instrument]))
        samples = [wavfile.read(sample) for sample in samples]
        longest_sample_len = max([len(sample[1]) for sample in samples])
        shifts = [0] + [random.randint(0, max_chunks_shift) for _ in range(instruments_amount - 1)]
        for chunk_id in range(longest_sample_len // CHUNK_SIZE):
            result = [0] * len(INSTRUMENTS)
            chunk_sum = np.zeros(CHUNK_SIZE)
            for instrument_id, instrument in enumerate(instruments):
                chunk = samples[instrument_id][1][(shifts[instrument_id] + chunk_id) * CHUNK_SIZE:(shifts[
                                                                                                       instrument_id] + chunk_id + 1) * CHUNK_SIZE]
                if len(chunk) < CHUNK_SIZE:
                    continue
                if max(chunk) / 32768 > SILENT_CHUNK_THRESHOLD:
                    result[INSTRUMENTS.index(instrument)] = 1
                chunk_sum += chunk
            cnt = sum(result)
            if cnt > max_instruments or len(data[cnt]) >= data_amounts[cnt]:
                continue
            chunk_sum /= max_instruments
            chunk_sum /= 32768
            match data_type:
                case 'sample':
                    result_data = chunk_sum
                case 'fft':
                    result_data = np.fft.fft(chunk_sum)[:CHUNK_SIZE // 2]
                    result_data = np.abs(result_data)
                case 'harmonics':
                    result_data = note_to_harmonics(44100, chunk_sum, 20)
                case _:
                    raise ValueError('Unknown data type')

            data[cnt].append([result_data, result])

    print('Data size: ' + str(sum(len(data[i]) for i in range(max_instruments + 1))))

    # save data to file data.data
    with open(data_filename, 'w') as f:
        for cnt in range(len(data)):
            for i in range(len(data[cnt])):
                f.write(" ".join(map(str, data[cnt][i][0])) + "|" + " ".join(map(str, data[cnt][i][1])) + '\n')


def load_data(filename, train_test_ratio=0.8):
    data = [[] for _ in range(len(INSTRUMENTS) + 1)]

    with open(filename, 'r') as f:
        for line in f:
            features, answers = line.split('|')
            print(line)
            print(line.split('|'))
            print(line.count('|'))
            print(features)
            print(answers)

            features = [float(x) for x in features.split()]
            answers = [int(x) for x in answers.split()]
            data_line = [features, answers]
            cnt = sum(data_line[1])
            data[cnt].append(data_line)

    train_data = [[] for _ in range(len(INSTRUMENTS) + 1)]
    test_data = [[] for _ in range(len(INSTRUMENTS) + 1)]

    for i in range(len(INSTRUMENTS) + 1):
        train_data[i] = data[i][:int(len(data[i]) * train_test_ratio)]
        test_data[i] = data[i][int(len(data[i]) * train_test_ratio):]

    train_data = [x for y in train_data for x in y]
    test_data = [x for y in test_data for x in y]

    random.shuffle(train_data)
    random.shuffle(test_data)

    return train_data, test_data
