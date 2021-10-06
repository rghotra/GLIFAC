import tensorflow as tf
import numpy as np

def get_layer_output(model, index, X):
    temp = tf.keras.Model(inputs=model.inputs, outputs=model.layers[index].output)
    return temp.predict(X)


def get_filter_activations(model, index, X):
    temp = tf.keras.Model(inputs=model.inputs, outputs=model.layers[index].output)
    
    filter_activations = temp.predict(X)
    filter_activations = tf.transpose(filter_activations, [2, 0, 1])
    filter_activations = tf.reshape(filter_activations, (filter_activations.shape[0], -1))

    return filter_activations


def local_attention(model, c_index, mha_index, X, thresh=0.1):
    feature_maps = get_layer_output(model, c_index, X)
    num_filters = feature_maps.shape[2]
    o, att_maps = get_layer_output(model, mha_index, X)
    att_maps = head_concat(att_maps, axis=1)
    filter_activations = get_filter_activations(model, index, X)

    position_interactions = get_position_interactions(att_maps, thresh)

    # convert to filter interactions

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