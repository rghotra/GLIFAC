# imports -----------------------------------------------------------------------------------------------------

import numpy as np
import tensorflow as tf
import sklearn

import tfomics
from tfomics import moana

import os, shutil
from six.moves import cPickle

import h5py, io
import requests as rq

import utils
import glifac

import ushuffle

import matplotlib.pyplot as plt

# load data -------------------------------------------------------------------------------------------------------------

def load_dataset(file_path):
    
    global X, Y, x_train, y_train, x_valid, y_valid, x_test, y_test, L

    with h5py.File(file_path, 'r') as dataset:
    	X = np.array(dataset['X'])
    	Y = np.array(dataset['Y'])
    	L = np.array(dataset['L'])

    train = int(len(X) * 0.7)
    valid = train + int(len(X) * 0.1 )
    test = valid + int(len(X) * 0.2)

    x_train = X[:train]
    x_valid = X[train:valid]
    x_test = X[valid:test]

    y_train = Y[:train]
    y_valid = Y[train:valid]
    y_test = Y[valid:test]
    
    
def load_gia_sequences(file_path):
    
    global indep, inter
    
    with h5py.File(file_path, 'r') as dataset:
        indep = np.array(dataset['independent'])
        inter = np.array(dataset['interactions'])
    
    
    
import logomaker
import pandas as pd

def plot_filters(W, fig, num_cols=8, alphabet='ACGT', names=None, fontsize=12):
    """plot 1st layer convolutional filters"""

    num_filter, filter_len, A = W.shape
    num_rows = np.ceil(num_filter/num_cols).astype(int)

    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    for n, w in enumerate(W):
        ax = fig.add_subplot(num_rows,num_cols,n+1)

        # Calculate sequence logo heights -- information
        I = np.log2(4) + np.sum(w * np.log2(w+1e-7), axis=1, keepdims=True)
        logo = I*w

        # Create DataFrame for logomaker
        counts_df = pd.DataFrame(data=0.0, columns=list(alphabet), index=list(range(filter_len)))
        for a in range(A):
            for l in range(filter_len):
                counts_df.iloc[l,a] = logo[l,a]

        logomaker.Logo(counts_df, ax=ax)
        ax = plt.gca()
        ax.set_ylim(0,2)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        plt.xticks([])
        plt.yticks([])
        if names is not None:
            plt.ylabel(names[n], fontsize=fontsize)    






# pipeline -------------------------------------------------------------------------------------------------------------
    
def run_pipeline(model, path, baseline, category, variant, trial, motifs, batch_size=200, epochs=100, root_tomtom=False, concat=np.max, rand_frac=0.5, symmetrize=glifac.absmaxND):
    
    jaspar_ids, motif_names, expecteds = motifs
    
    global x_train, y_train, x_valid, y_valid, x_test, y_test, indep, inter

    # Create directories
    owd = os.getcwd()
    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)
    model_dir = os.path.abspath(f'{baseline}/models/{category}/model-{variant}')
    motif_dir = os.path.abspath(f'{baseline}/motifs/{category}/model-{variant}')
    tomtom_dir = os.path.abspath(f'{baseline}/tomtom/{category}/model-{variant}')
    stats_dir = os.path.abspath(f'{baseline}/stats/{category}/model-{variant}')
    logs_dir = os.path.abspath(f'{baseline}/history/{category}/model-{variant}')
    ppms_dir = os.path.abspath(f'{baseline}/ppms/{category}/model-{variant}')
    glifac_dir = os.path.abspath(f'{baseline}/glifac/{category}/model-{variant}')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(motif_dir):
        os.makedirs(motif_dir)
    if not os.path.exists(tomtom_dir):
        os.makedirs(tomtom_dir)
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    if not os.path.exists(ppms_dir):
        os.makedirs(ppms_dir)
    if not os.path.exists(glifac_dir):
        os.makedirs(glifac_dir)
    
    model_dir += f'/trial-{trial}/weights'
    motif_dir += f'/trial-{trial}.txt'
    tomtom_dir += f'/trial-{trial}'
    stats_dir += f'/trial-{trial}.npy'
    logs_dir += f'/trial-{trial}.pickle'
    ppms_dir += f'/trial-{trial}.pdf'
    glifac_dir += f'/trial-{trial}.pdf'
    
    if os.path.exists(tomtom_dir):
        shutil.rmtree(tomtom_dir)
    
    # get important indices
    lays = [type(i) for i in model.layers]
    c_index = lays.index(tf.keras.layers.MaxPool1D)
    mha_index = lays.index(tfomics.layers.MultiHeadAttention)
    
    # train model ------------------------------------------------------------------------------------------------------
    
    auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
    aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
    model.compile(
        tf.keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=[auroc, aupr]
    )

    lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patient=5, verbose=1, min_lr=1e-7, mode='min')
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min', restore_best_weights=True)
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_valid, y_valid), callbacks=[lr_decay, early_stop], verbose=2, batch_size=batch_size)
    
    model.save_weights(model_dir) # save model weights
    
    with open(logs_dir, 'wb') as handle:
        cPickle.dump(history.history, handle) # save model history
    
    # evaluate performance --------------------------------------------------------------------------------------------
    
    loss, auc_roc, auc_pr = model.evaluate(x_test, y_test)
    
    # filter interpretability -----------------------------------------------------------------------------------------
    
    # get ppms
    ppms = utils.get_ppms(model, x_test)
    moana.meme_generate(ppms, output_file=motif_dir) # save filter PPMs
    print('generated PPMs')
    
    # filter interpretablity
    motif_database = path[:path.index('GLIFAC')] + 'GLIFAC/Datasets/motif_database.txt'
    utils.tomtom(motif_dir, tomtom_dir, database=motif_database, root=root_tomtom) # save tomtom files
    names = ['ELF1', 'SIX3', 'ESR1', 'FOXN', 'CEBPB', 'YY1', 'GATA1', 'IRF1', 'SP1', 'NFIB', 'TEAD', 'TAL1']

    match_fraction, match_any, filter_match, filter_qvalue, min_qvalue, num_counts, coverage = utils.get_tomtom_stats(tomtom_dir + '/tomtom.tsv', 32)
    filter_matches = np.array(filter_match)
    tomtom_tpr = match_fraction
    tomtom_fpr = match_any - match_fraction
    tomtom_cov = coverage
    qvals = filter_qvalue
    
    print('TomTom TPR: ', tomtom_tpr)
    print('TomTom FPR: ', tomtom_fpr)
    print('Motif coverage: ', tomtom_cov)
    print('completed TomTom analysis')
    
    # hierachical clustering
    
    order2 = np.where(filter_matches == '')[0]
    order1 = np.where(filter_matches != '')[0]
    order1 = order1[np.argsort(filter_matches[order1])]
    order = np.hstack([order1, order2])
    
    conv_weights = model.layers[1].get_weights()[0]
    conv_weights = conv_weights.transpose()[order].transpose()
    model.layers[1].set_weights([conv_weights])
    filter_labels = filter_matches[order]

    ppm_size = 25
    filter_ppms = []
    for i in range(len(ppms)):
        padded = np.vstack([ppms[i], np.zeros((ppm_size-len(ppms[i]), 4))+0.25])
        filter_ppms.append(padded)
    filter_ppms = np.array(filter_ppms)[order]
    
    information = []
    for i in range(len(filter_ppms)):
        I = np.log2(4) + np.sum(filter_ppms[i] * np.log2(filter_ppms[i] + 1e-7), axis=1)
        information.append(I)
    information = np.sum(information, axis=1)

    empty_filters = np.where(information < 0.5)[0]
    
    print('completed clustering')
    
    # glifac ----------------------------------------------------------------------------------------
    
    sample = x_test[:5000]

    feature_maps = utils.get_layer_output(model, c_index, sample)
    num_filters = feature_maps.shape[2]

    o, att_maps = utils.get_layer_output(model, mha_index, x_test)
    att_maps = concat(att_maps, axis=1)
    for i in range(len(att_maps)):
        np.fill_diagonal(att_maps[i], 0)
    all_attention_values = np.reshape(att_maps, -1)

    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    correlation_maps = []
    correlation_aurocs = []
    correlation_auprs = []
    correlation_snrs = []
    for i in range(len(thresholds)):

        if thresholds[i] > np.sort(all_attention_values)[-2]:
            correlation_map = np.zeros((feature_maps.shape[2], feature_maps.shape[2]))
            corr_tpr = 0
            corr_fpr = 0
            corr_cov = 0
        else:
            correlation_map = glifac.correlation_matrix(model, c_index, mha_index, sample, thresh=thresholds[i], random_frac=rand_frac, limit=250000, head_concat=concat, symmetrize=symmetrize)
            correlation_map = glifac.remove_self_interactions(correlation_map, filter_labels)
        correlation_maps.append(correlation_map)
        
        corr_pr, corr_roc, corr_snr = glifac.glifac_stats(correlation_map, filter_labels, expecteds)

        correlation_aurocs.append(corr_roc)
        correlation_auprs.append(corr_pr)
        correlation_snrs.append(corr_snr)
        
        print(thresholds[i], ' | AUROC:', corr_roc, ' | AUPR:', corr_pr, ' | SNR:', corr_snr)

    correlation_maps = np.array(correlation_maps)
    correlation_aurocs = np.array(correlation_aurocs)
    correlation_auprs = np.array(correlation_auprs)
    correlation_snrs = np.array(correlation_snrs)
    print('finished correlation interpretability')
    
    # save all statistics
    
    stats = [correlation_maps, correlation_aurocs, correlation_auprs, correlation_snrs, tomtom_tpr, tomtom_fpr, tomtom_cov, qvals, auc_roc, auc_pr, filter_labels]
    
    np.save(stats_dir, stats) # save stats
    print('saved statistics')
    
    
    
    
    
    ppms = utils.get_ppms(model, x_test)
    ppm_size = 25
    filter_ppms = []
    for i in range(len(ppms)):
        padded = np.vstack([ppms[i], np.zeros((ppm_size-len(ppms[i]), 4))+0.25])
        filter_ppms.append(padded)
    ppms = np.array(filter_ppms)
    fig = plt.figure(figsize=(25,4))
    plot_filters(ppms, fig, num_cols=8, names=filter_labels, fontsize=14)
    fig.savefig(ppms_dir, format='pdf', dpi=200, bbox_inches='tight')
    print('saved filters')
    
    
    fig = plt.figure(figsize=(15,15))
    ax = fig.subplots()
    glifac.plot_glifac(ax, correlation_maps[1], filter_labels, vmin=-0.5, vmax=0.5)
    fig.savefig(glifac_dir, format='pdf', dpi=300, box_inches='tight')
    print('saved glifac')
    
    os.chdir(owd)
    
    






































