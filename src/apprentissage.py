import glob
from sys import argv
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import shutil
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

verbose = True
k1 = int(argv[1])

lesMfcc = np.empty(shape=(0, 13), dtype=float)  # array of all MFCC from all sounds
dimSons = []  # nb of mfcc per file
listSons = glob.glob("../resources/base_samples/*.wav")
labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

for s in listSons:
    if verbose:
        print("###", s, "###")
    (rate, sig) = wav.read(s)
    mfcc_feat = mfcc(sig, rate)
    if verbose:
        print("MFCC: ", mfcc_feat.shape)
    dimSons.append(mfcc_feat.shape[0])
    lesMfcc = np.append(lesMfcc, mfcc_feat, axis=0)

# BOW initialization
bows = np.empty(shape=(0, k1), dtype=int)

# everything ready for the 1st k-means
try:
    with open("../resources/kmeans1.txt", "rb") as input:
        kmeans1 = pickle.load(input)
except EOFError:
    kmeans1 = KMeans(n_clusters=k1, random_state=0).fit(lesMfcc)

if verbose:
    print("result of kmeans 1", kmeans1.labels_)

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

print(bows)

#cr´eation d'un objet de regression logistique
logisticRegr = LogisticRegression()
#apprentissage
logisticRegr.fit(bows, labels)
#calcul des labels pr´edits
labelsPredicted = logisticRegr.predict(bows)
#calcul et affichage du score
score = logisticRegr.score(bows, labels)
print("train score = ", score)


#sauvegarde de l'objet
with open('sauvegarde.logr', 'wb') as output:
    pickle.dump(logisticRegr, output, pickle.HIGHEST_PROTOCOL)
#chargement de l'objet
with open('sauvegarde.logr', 'rb') as input:
    logisticRegr = pickle.load(input)

print(score)
print(confusion_matrix(labels, labelsPredicted))
