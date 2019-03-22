import glob
from sys import argv
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import shutil
from sklearn.cluster import KMeans
import numpy as np
import pickle

verbose = True
k1 = int(argv[1])
k2 = int(argv[2])

lesMfcc = np.empty(shape=(0, 13), dtype=float)  # array of all MFCC from all sounds
dimSons = []  # nb of mfcc per file
listSons = glob.glob("../resources/test_samples/*.wav")

for s in listSons:
    if verbose:
        print("###", s, "###")
    (rate, sig) = wav.read(s)
    mfcc_feat = mfcc(sig, rate)
    if verbose:
        print("MFCC: ", mfcc_feat.shape)
    dimSons.append(mfcc_feat.shape[0])
    lesMfcc = np.append(lesMfcc, mfcc_feat, axis=0)

try:
    with open("../resources/kmeans1.txt", "rb") as input:
        kmeans1 = pickle.load(input)
except EOFError:
    exit(-1)

bows = np.empty(shape=(0, k1), dtype=int)

kmeans1new = kmeans1.predict(lesMfcc)

i = 0
for nb in dimSons:  # for each sound (file)
    tmpBow = [0] * k1
    j = 0
    while j < nb:  # for each MFCC of this sound (file)
        tmpBow[kmeans1new[i]] += 1
        j += 1
        i += 1
    copyBow = tmpBow.copy()
    bows = np.append(bows, [copyBow], 0)

print(bows)

try:
    with open("../resources/kmeans2.txt", "rb") as input:
        kmeans2 = pickle.load(input)
except EOFError:
    exit(-1)


print(kmeans2.predict(bows))
