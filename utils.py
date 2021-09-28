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

def get_queries_keys(model, index, fmaps):
    mha = model.layers[index]
    
    q = mha.wq(fmaps)
    k = mha.wk(fmaps)
    
    q = mha.split_heads(q, fmaps.shape[0], fmaps.shape[1])
    k = mha.split_heads(k, fmaps.shape[0], fmaps.shape[1])
    
    return q, k

def get_attention_maps(q, k, concat=tf.math.reduce_max):
    
    o, att_maps = scaled_dot_product_attention(q, k, k)
    att_maps = concat(att_maps, axis=-3)
    
    return att_maps

def get_position_interactions(att_maps, threshold=0.1, limit=100000):
    position_interactions = np.array(np.where(att_maps >= threshold))
    position_interactions = [position_interactions[[0]].transpose(), position_interactions[[1, 2]].transpose()]
    
    permutation = np.random.permutation(len(position_interactions[0]))
    position_interactions = [position_interactions[0][permutation], position_interactions[1][permutation]]
    position_interactions = [position_interactions[0][:limit], position_interactions[1][:limit]]
    
    return position_interactions

def get_filter_activations(model, index, X):
    temp = tf.keras.Model(inputs=model.inputs, outputs=model.layers[index].output)
    
    filter_activations = temp.predict(X)
    filter_activations = tf.transpose(filter_activations, [2, 0, 1])
    filter_activations = tf.reshape(filter_activations, (filter_activations.shape[0], -1))
    
    return filter_activations

def get_filter_interactions(feature_maps, position_interactions, filter_activations):
    
    num_filters = feature_maps.shape[2]
    adj_fmaps = feature_maps - filter_activations
    adj_fmaps = np.array(adj_fmaps > 0).astype(int)
    vector_interactions = adj_fmaps[position_interactions]

    # actual filter values
    first_positions = vector_interactions[:,0]
    second_positions = vector_interactions[:,1]

    first_positions = np.expand_dims(first_positions, axis=1)
    first_positions = np.repeat(first_positions, second_positions.shape[1], axis=1) # repeat columns

    second_positions = np.expand_dims(second_positions, axis=2)
    second_positions = np.repeat(second_positions, first_positions.shape[1], axis=2) # repeat rows

    meshgrid = np.array([first_positions, second_positions]).transpose([1, 0, 2, 3])
    meshgrid = meshgrid.transpose([0, 3, 2, 1]).reshape((meshgrid.shape[0], -1, 2))

    # filter place holders
    first_filters = np.repeat(np.expand_dims(np.arange(num_filters), axis=0), vector_interactions.shape[0], axis=0)
    second_filters = np.repeat(np.expand_dims(np.arange(num_filters), axis=0), vector_interactions.shape[0], axis=0)

    first_filters = np.expand_dims(first_filters, axis=1)
    first_filters = np.repeat(first_filters, second_filters.shape[1], axis=1) # repeat columns

    second_filters = np.expand_dims(second_filters, axis=2)
    second_filters = np.repeat(second_filters, first_filters.shape[1], axis=2) # repeat rows

    filter_meshgrid = np.array([first_filters, second_filters]).transpose([1, 0, 2, 3])
    filter_meshgrid = filter_meshgrid.transpose([0, 3, 2, 1]).reshape((filter_meshgrid.shape[0], -1, 2))

    # mask inactive filters
    mask = np.ones(meshgrid.shape, dtype=bool)
    mask[np.where(meshgrid == 0)[:2]] = False
    mask = mask.transpose()[0].transpose()

    filter_interactions = filter_meshgrid[mask]
    
    return filter_interactions.astype(int)
    

def get_clusters(correlation_map, filter_matches, threshold):
    
    dissimilarity = 1 - abs(correlation_map)
    Z = linkage(squareform(dissimilarity), 'complete')

    labels = fcluster(Z, threshold, criterion='distance')
    labels_order = np.argsort(labels)
    
    filter_labels = [f'{i}-{filter_matches[i]}' for i in labels_order]
    
    groups = [[] for i in range(np.max(labels))]
    for i in range(len(labels)):
        groups[labels[i]-1].append(i)
        
    return Z, labels, labels_order, filter_labels, groups
    
def hierarchichal_clustering(pooled_fmaps, threshold, filter_matches, method='activation', tomtom_dir=None, concat=np.amax):
    clustered_fmaps = pooled_fmaps.reshape((-1, pooled_fmaps.shape[2])).transpose()

    if method == 'activation':
        correlation_map = get_correlations(clustered_fmaps, clustered_fmaps)
        correlation_map = np.nan_to_num(correlation_map)
        np.fill_diagonal(correlation_map, 1)
    elif method == 'tomtom':
        df = pd.read_csv(tomtom_dir, delimiter='\t')[:-3]
        sources = df['Query_ID'].to_numpy()
        targets = df['Target_ID'].to_numpy()
        qvalues = df['q-value'].to_numpy()

        sources = np.array([int(i[6:]) for i in sources])
        targets = np.array([int(i[6:]) for i in targets])

        correlation_map = np.zeros((len(filter_matches), len(filter_matches)))
        correlation_map[sources, targets] = 1
        
        correlation_map = np.nan_to_num(correlation_map)
        np.fill_diagonal(correlation_map, 1)
        correlation_map = (correlation_map + correlation_map.transpose())/2
        
    Z, labels, labels_order, filter_labels, groups = get_clusters(correlation_map, filter_matches, threshold)
    
    tp_fmaps = pooled_fmaps.transpose()
    clustered_fmaps = []
    group_names = []
    for i in range(len(groups)):
        fmap = concat(tp_fmaps[groups[i]], axis=0)
        clustered_fmaps.append(fmap)

        names = filter_matches[groups[i]]
        u, counts = np.unique(names, return_counts=True)
        group_names.append(u[0])
    clustered_fmaps = np.array(clustered_fmaps).transpose()
    
    return labels, labels_order, Z, groups, group_names, clustered_fmaps

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

def get_local_attention(model, pool_index, mha_index, sample, exp_ints, thresh=0.1):
    
    feature_maps = get_layer_output(model, pool_index, sample)
    num_filters = feature_maps.shape[2]

    mha_input = get_layer_output(model, mha_index-1, sample)
    q, k = get_queries_keys(model, mha_index, mha_input)
    att_maps = get_attention_maps(q, k, concat=tf.math.reduce_max)
    
    pos_ints = np.array(np.where(att_maps >= thresh)).transpose()
    last_cols = pos_ints[:,[1, 2]]
    last_cols = np.sort(last_cols, axis=-1)
    pos_ints[:,[1, 2]] = last_cols
    
    total_matches = 0
    for i in range(len(exp_ints)):
        trunc = np.where(pos_ints[:,0] == exp_ints[i][0])[0]
        if len(trunc) == 0:
            continue
        matches = np.sum((pos_ints[trunc[0]:trunc[-1]+1] == exp_ints[i]).astype(int), axis=1)
        matches = np.where(matches == 3)[0]
        total_matches += len(matches)
    local_cov = total_matches/exp_ints.shape[0]

    total_matches = 0
    for i in range(len(pos_ints)):
        matches = np.sum((exp_ints == pos_ints[i]).astype(int), axis=1)
        matches = np.where(matches == 3)[0]
        total_matches += len(matches)
    local_tpr = total_matches/len(pos_ints)
    local_fpr = 1 - local_tpr
    
    return local_tpr, local_fpr, local_cov