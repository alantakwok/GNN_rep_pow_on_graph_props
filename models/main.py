import pickle

import GraphConvNet as gcn


datapath = '.'
units = 1
activation = 'sigmoid'
bias = False
skip = True
nepochs = 2000
ep_split = 50
outdir = '.'
ep_split = 200
moment = 1


dataset = pickle.load(open(datapath, 'rb'))
Adjacencies = dataset['adjacency']
h = dataset['h']
indices = dataset['indicies']
labels = dataset['labels']


gcn_model = gcn.MultiGCN(
    input_shape=dataset['moments'][1][0].shape,
    units=units,
    activation=activation,
    skip=skip,
    dense_kws={'use_bias': bias},
    GCN_kws={'use_bias': bias}
)

for i in range(int(nepochs // ep_split)):
    # first input is the adjacency matrix, second input is ones, third input is moments
    gcn_model.train([Adjacencies[indices], h[indices]], [labels[indices]], epochs=ep_split)
    gcn_model.save('{}/gcn_model-units{}-moments{}-Act_{}-skip{}.pkl'.format(
        outdir, units, moment, activation, skip))