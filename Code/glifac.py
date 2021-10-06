import numpy as np
import sklearn
import tensorflow as tf



def absmaxND(a, axis=None):
    amax = np.max(a, axis)
    amin = np.min(a, axis)
    return np.where(-amin > amax, amin, amax)


def get_layer_output(model, index, X):
    temp = tf.keras.Model(inputs=model.inputs, outputs=model.layers[index].output)
    return temp.predict(X)

def pearsonr(vector1, vector2):
    m1 = np.mean(vector1)
    m2 = np.mean(vector2)
    
    diff1 = vector1 - m1
    diff2 = vector2 - m2
    
    top = np.sum(diff1 * diff2)
    bottom = np.sum(np.power(diff1, 2)) * np.sum(np.power(diff2, 2))
    bottom = np.sqrt(bottom)
    
    return top/bottom





def plot_glifac(ax, correlation_matrix, filter_labels, vmin=-0.5, vmax=0.5):
    ax.set_xticks(list(range(len(filter_labels))))
    ax.set_yticks(list(range(len(filter_labels))))
    ax.set_xticklabels(filter_labels, rotation=90)
    ax.set_yticklabels(filter_labels)
    c = ax.imshow(correlation_matrix, cmap='bwr_r', vmin=vmin, vmax=vmax)
    return ax, c

def glifac_stats(correlation_matrix, filter_labels, expecteds):
    
    ind = np.array(np.tril_indices(len(correlation_matrix), k=-1)).transpose()
    motif_interactions = filter_labels[ind]
    filter_interactions_values = correlation_matrix[ind.transpose().tolist()]
    mask = ~(motif_interactions.transpose() == motif_interactions.transpose()[::-1])[0]
    motif_interactions = motif_interactions[mask]
    filter_interactions_values = filter_interactions_values[mask]
    motif_interactions.sort()

    matches = []
    for i in range(len(expecteds)):
        match = (expecteds[i] == motif_interactions).astype(int).transpose()
        match = match[0] * match[1]
        matches.append(match)
    TPs = np.amax(matches, axis=0).astype(bool)

    trues = filter_interactions_values[TPs]
    falses = filter_interactions_values[~TPs]

    k = 100
    signal = np.mean(trues)
    noise = np.sort(falses)[::-1][:k]
    noise = noise[np.where(noise > 0)]
    noise = np.mean(noise)
    snr = signal/(noise + np.finfo(float).eps)

    y_pred = np.hstack([trues, falses])
    y_true = np.hstack([np.ones(trues.shape), np.zeros(falses.shape)])
    precision, recall, ts = sklearn.metrics.precision_recall_curve(y_true, y_pred)
    pr = sklearn.metrics.auc(recall, precision)
    specificity, sensitivity, ts = sklearn.metrics.roc_curve(y_true, y_pred)
    roc = sklearn.metrics.auc(specificity, sensitivity)
    
    return roc, pr, snr


def correlation_matrix(model, c_index, mha_index, X, thresh=0.1, random_frac=0.5, limit=None, head_concat=np.max, symmetrize=absmaxND):
    
    """
    * model                  trained tensorflow model
    * c_index                index of the convolutoinal layer (after pooling)
    * mha_index              index of multi-head attention layer
    * X                      test sequences
    * thresh                 attention threshold
    * random_frac            proportion of negative positions in the set of position interactions
    * limit                  maximum number of position interactions processed; sometimes needed to avoid resource exhaustion
    * head_concat            function for concatenating heads; e.g. np.max, np.mean
    * symmetrize             function for symmetrizing the correlation matrix across diagonal
    """
    
    assert 0 <= random_frac < 1
    
    feature_maps = get_layer_output(model, c_index, X)
    o, att_maps = get_layer_output(model, mha_index, X)
    att_maps = head_concat(att_maps, axis=1)
    
    position_interactions = get_position_interactions(att_maps, thresh)
    num_rands = int(random_frac/(1-random_frac))
    random_interactions = [np.random.randint(len(att_maps), size=(num_rands, 1)), np.random.randint(att_maps.shape[1], size=(num_rands, 2))]
    position_pairs = [np.vstack([position_interactions[0], random_interactions[0]]), np.vstack([position_interactions[1], random_interactions[1]])]
    if limit is not None:
        permutation = np.random.permutation(len(position_pairs[0]))
        position_pairs = [position_pairs[0][permutation], position_pairs[1][permutation]]
        position_pairs = [position_pairs[0][:limit], position_pairs[1][:limit]]
    
    filter_interactions = feature_maps[position_pairs].transpose([1, 2, 0])
    correlation_matrix = correlation(filter_interactions[0], filter_interactions[1])
    if symmetrize is not None:
        correlation_matrix = symmetrize(np.array([correlation_matrix, correlation_matrix.transpose()]), axis=0)
    correlation_matrix = np.nan_to_num(correlation_matrix)
    
    return correlation_matrix

    
def get_position_interactions(att_maps, threshold=0.1):
    position_interactions = np.array(np.where(att_maps >= threshold))
    position_interactions = [position_interactions[[0]].transpose(), position_interactions[[1, 2]].transpose()]
    
    return position_interactions
    
    
def correlation(set1, set2):
    combinations = np.array(np.meshgrid(np.arange(len(set1)), np.arange(len(set2)))).reshape((2, -1))[::-1]
    vector_mesh = [set1[combinations[0]], set2[combinations[1]]]
    vector_mesh = np.array(vector_mesh).transpose([1, 0, 2])
    
    correlations = []
    for i in range(len(vector_mesh)):
        r = pearsonr(vector_mesh[i][0], vector_mesh[i][1])
        correlations.append(r)
    correlations = np.array(correlations).reshape((len(set1), len(set2)))
    
    return correlations


def remove_self_interactions(correlation_matrix, filter_labels):
    correlation_map = correlation_matrix.copy()
    num_filters = len(filter_labels)
    
    ind = np.array(np.meshgrid(np.arange(num_filters), np.arange(num_filters))).transpose([1, 2, 0])
    motif_interactions = filter_labels[ind]
    correlation_map[np.where(motif_interactions[:,:,0] == motif_interactions[:,:,1])] = 0
    
    return correlation_map

