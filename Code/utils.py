import tensorflow as tf
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, leaders
from scipy.spatial.distance import squareform
import pandas as pd

from tfomics import moana
from tfomics.layers import MultiHeadAttention, scaled_dot_product_attention

import requests as rq
import io

import subprocess
import glifac

elf = ['MA0473.3']
six = ['MA0631.1']
tal = ['MA0048.2', 'MA0048.1']
gata = ['MA0035.4', 'MA0035.3']
foxn = ['MA1489.1']
cebpb = ['MA0466.1', 'MA0466.2']
nfib = ['MA1643.1']
yy1 = ['MA0095.1', 'MA0095.2']
tead = ['MA0090.1', 'MA0809.1', 'MA1121.1']
esr = ['MA0112.3']
irf = ['MA0652.1', 'MA0653.1']
sp1 = ['MA0079.1', 'MA0079.2', 'MA0079.3', 'MA0079.4']

tcp1 = ['MA1284.1']
pax4 = ['MA0068.1']

motif_names = ['ELF1', 'SIX3', 'ESR1', 'FOXN', 'CEBPB', 'YY1', 'GATA1', 'IRF1', 'SP1', 'NFIB', 'TEAD', 'TAL1']
motifs = [elf, six, esr, foxn, cebpb, yy1, gata, irf, sp1, nfib, tead, tal]

def normalize(x):
    x_min = tf.math.reduce_min(tf.reshape(x, [-1]))
    x_translated = tf.subtract(x, x_min)

    x_max = tf.math.reduce_max(tf.reshape(x_translated, [-1]))
    x_scaled = tf.divide(x_translated, x_max)
    
    return x_scaled

def absmaxND(a, axis=None):
    amax = np.max(a, axis)
    amin = np.min(a, axis)
    return np.where(-amin > amax, amin, amax)

def get_ppms(model, x_test):
    index = [i.name for i in model.layers].index('conv_activation')
    
    ppms = moana.filter_activations(x_test, model, layer=index, window=20,threshold=0.5)
    ppms = moana.clip_filters(ppms, threshold=0.5, pad=3)
    
    return ppms

def tomtom(motif_dir, output_dir, database='motif_database.txt', thresh=0.25, root=False):
	if root:
		t = '~/meme/bin/tomtom'
	else:
		t = 'tomtom'
	cmd = f'{t} -evalue -thresh {str(thresh)} -o {output_dir} {motif_dir} {database}'
	subprocess.call(cmd, shell=True)
    
def get_qvalues(filter_hits, filter_qvals, motif_names):
    motifs = {}
    for name in motif_names:
        motifs[name] = 1
    motifs[''] = 1
    
    for hit, qval in zip(filter_hits, filter_qvals):
        if qval < motifs[hit]:
            motifs[hit] = qval
            
    qvalues = []
    for motif_name in motifs.keys():
        qvalues.append(motifs[motif_name])
    qvalues = np.array(qvalues)
    index = np.where(qvalues[:-1] == 1)[0]
    coverage = 1 - len(index)/len(motif_names)
    return qvalues[:-1], coverage
    
def get_tomtom_stats(tomtom_dir, num_filters=32):
    
    match_fraction, match_any, filter_match, filter_qvalue, min_qvalue, num_counts = moana.match_hits_to_ground_truth(tomtom_dir, motifs, motif_names, num_filters)
    
    qvalues, coverage = get_qvalues(filter_match, filter_qvalue, motif_names)
    
    return match_fraction, match_any, filter_match, qvalues, min_qvalue, num_counts, coverage

def get_layer_output(model, index, x):
    temp = tf.keras.Model(inputs=model.inputs, outputs=model.layers[index].output)
    
    return temp.predict(x)


def hierarchichal_clustering(model, c_index, X, threshold):
    
    pooled_fmaps = get_layer_output(model, c_index, X)
    
    clustered_fmaps = pooled_fmaps.reshape((-1, pooled_fmaps.shape[2])).transpose()

    correlation_map = correlation(clustered_fmaps, clustered_fmaps)
    correlation_map = np.nan_to_num(correlation_map)
    np.fill_diagonal(correlation_map, 1)
    
    dissimilarity = 1 - abs(correlation_map)
    Z = linkage(squareform(dissimilarity), 'complete')
    
    labels = fcluster(Z, threshold, criterion='distance')
    labels_order = np.argsort(labels)
    
    return labels_order

def get_motif_interactions(filter_interactions, filter_matches):
    filter_matches = np.append(filter_matches, 'None')
    motif_interactions = filter_matches[filter_interactions]
    motif_interactions.sort()
    return motif_interactions

def get_match_frac(motif_interactions, expected):
    if len(motif_interactions) == 0:
        return 0
    matches = (motif_interactions == expected).astype(int)
    counts = len(np.where(np.sum(matches, axis=1) == len(expected))[0])
    match_frac = counts/len(motif_interactions)
    return match_frac

def get_interaction_stats(motif_interactions, expecteds):
    tpr = 0
    for i in range(len(expecteds)):
        tpr += get_match_frac(motif_interactions, expecteds[i])
    
    fpr = 1 - tpr
    
    cov = 0
    for i in range(len(expecteds)):
        if str(expecteds[i])[1:-1] in str(np.array(motif_interactions).tolist()):
            cov += 1
    cov /= len(expecteds)
    
    return tpr, fpr, cov