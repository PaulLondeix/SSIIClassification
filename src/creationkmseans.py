import glob
from sys import argv
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import shutil
from sklearn.cluster import KMeans
import numpy as np
import pickle

# usage: python3 recosons2018.py k1 k2 verbose
# ATTENTION: les noms de fichiers ne doivent comporter ni - ni espace

# sur ligne de commande: les 2 parametres de k means puis un param de verbose
k1 = int(argv[1])
k2 = int(argv[2])

if argv[3] == "True":
    verbose = True
else:
    verbose = False

listSons = glob.glob("../resources/base_samples/*.wav")

lesMfcc = np.empty(shape=(0, 13), dtype=float)  # array of all MFCC from all sounds
dimSons = []  # nb of mfcc per file

for s in listSons:
    if verbose:
        print("###", s, "###")
    (rate, sig) = wav.read(s)
    mfcc_feat = mfcc(sig, rate)
    if verbose:
        print("MFCC: ", mfcc_feat.shape)
    dimSons.append(mfcc_feat.shape[0])
    lesMfcc = np.append(lesMfcc, mfcc_feat, axis=0)

# everything ready for the 1st k-means
try:
    with open("../resources/kmeans1.txt", "rb") as input:
        kmeans1 = pickle.load(input)
except EOFError:
    kmeans1 = KMeans(n_clusters=k1, random_state=0).fit(lesMfcc)

if verbose:
    print("result of kmeans 1", kmeans1.labels_)

# BOW initialization
bows = np.empty(shape=(0, k1), dtype=int)

# writing the BOWs for second k-means
i = 0
for nb in dimSons:  # for each sound (file)
    tmpBow = [0] * k1
    j = 0
    while j < nb:  # for each MFCC of this sound (file)
        tmpBow[kmeans1.labels_[i]] += 1
        j += 1
        i += 1
    copyBow = tmpBow.copy()
    bows = np.append(bows, [copyBow], 0)
if verbose:
    print("nb of MFCC vectors per file : ", dimSons)
    print("BOWs : ", bows)

# ready for second k-means
try:
    with open("../resources/kmeans2.txt", "rb") as input:
        kmeans2 = pickle.load(input)
except EOFError:
    kmeans2 = KMeans(n_clusters=k2, random_state=0).fit(bows)
if verbose:
    print("result of kmeans 2", kmeans2.labels_)

# ecriture
with open("../resources/kmeans1.txt", "wb") as output:
    pickle.dump(kmeans1, output, pickle.HIGHEST_PROTOCOL)
with open("../resources/kmeans2.txt", "wb") as output:
    pickle.dump(kmeans2, output, pickle.HIGHEST_PROTOCOL)
