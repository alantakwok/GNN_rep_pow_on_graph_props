# %% generate random graphs and calculate graph properties
# import required libraries
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import random
import time

# define general graph parameters
n = 64  # total number of nodes
ncases = 2000  # total number of cases per model (n > len(model parameters))
p_ER = [round(random.uniform(0.10, 0.90), 2) for _ in range(6000)]  # probability for edge creation in ER graph
# k_WS = [5] # each node is joined with its k nearest neighbors in WS graph
# p_WS = [0.3] # probability of rewiring each edge in WS graph
m_BA = [random.randint(5, 50) for _ in
        range(6000)]  # number of edges to attach from a new node to existing nodes in BA graph
if ncases % 2 != 0 or ncases >= len(p_ER) or ncases >= len(m_BA):
    raise ValueError("Number of cases is invalid for the parameter sweep space!")
print(f'Parameters defined!')


# normalize a vector
normArr = lambda x: (x - np.mean(x)) / np.std(x)


# average path length function
def simple_path_props(G):
    # initialize average path length vector
    avg_path_len = []
    # initialize number of paths vector
    num_paths = []
    # compute average length of all simple paths between two nodes
    # start timer
    start = time.perf_counter()
    for i in range(1, n):
        print(f"source: {0}, target: {i}")
        # compute simple paths between two nodes
        paths = nx.all_simple_paths(G, source=0, target=i)
        # compute list of pairs of nodes
        path_len = []
        for path in map(nx.utils.pairwise, paths):
            pairs = list(path)
            # print(pairs)
            # print(len(pairs))
            path_len.append(len(pairs))
        avg_path_len.append(np.mean(path_len))
        num_paths.append(len(path_len))
        print(f"average path length: {avg_path_len[-1]}")
        print(f"number of paths: {num_paths[-1]}")
    # end timer
    end = time.perf_counter()
    # time taken to compute avg path length
    print(f'Time taken to compute simple path properties: {end-start} seconds')
    return avg_path_len, num_paths


def clear_csv(folder):
    # function to clear all csv files from a directory
    for file in os.listdir(folder):
        if file.endswith('.csv'):
            file_path = os.path.join(folder, file)
            try:
                os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')


# %% generate erdos-renyi graph and save properties as csv
# create directory to store graph properties and adjacency matrices
dir_prop = 'er_graph_prop/'
dir_adjc = 'er_graph_adjc/'
# clear all previous csv files from directory
clear_csv(dir_prop)
clear_csv(dir_adjc)

icase = 0  # count valid cases to store
pcase = 0  # counter for model generation parameter
while icase != ncases / 2:
    # ER graph generation
    G = nx.erdos_renyi_graph(n, p_ER[pcase])
    # save adjacency matrix to a csv file
    filename = dir_adjc + 'case_' + '{:04d}'.format(icase) + '.csv'
    np.savetxt(filename, nx.adjacency_matrix(G).todense(), delimiter=',')
    pcase += 1
    # check conditions
    if nx.is_connected(G):
        if icase != 0:
            for istore in range(len(store_graphs)):
                if nx.is_isomorphic(store_graphs[istore], G):
                    print(f'Isomorphic graphs found! Not a valid case.')
                    continue
        else:
            # if running first case then store graph and properties
            store_graphs = {icase: G}
    else:
        print(f'Disconnected graph found! Not a valid case.')
        continue
    # if all condition satisfies then store graph and properties
    print(f'Valid case found! Running case {icase}: p = {p_ER[pcase]}')
    store_graphs[icase] = G
    icase += 1

    # # show graph
    # fig1 = plt.figure()
    # nx.draw(G, with_labels=True)
    # plt.title('ER Graph')
    # # degree distribution
    # fig2 = plt.figure()
    # plt.plot(nx.degree_histogram(G))
    # plt.title('Degree Distribution')

    # page rank
    page_rank = list(nx.pagerank(G).values())
    norm_pr = normArr(page_rank)
    # clustering coefficient of each node
    clust_coeff = list(nx.clustering(G).values())
    norm_cc = normArr(clust_coeff)
    # degree centrality
    deg_cent = list(nx.degree_centrality(G).values())
    norm_dc = normArr(deg_cent)
    # average neighbor degree
    avg_neb_deg = list(nx.average_neighbor_degree(G).values())
    norm_and = normArr(avg_neb_deg)
    # eccentricity
    eccentry = list(nx.eccentricity(G).values())
    norm_ecc = eccentry / np.linalg.norm(eccentry)
    # shortest path length
    short_len = dict(nx.shortest_path_length(G))
    short_avg_len = [np.mean(list(short_len[i].values())) for i in range(n)]
    norm_sal = normArr(short_avg_len)

    # save each graph properties to a csv file
    filename = dir_prop + 'case_' + '{:04d}'.format(icase) + '.csv'
    # create field names for each property
    fields = ['page_rank', 'norm_pr', 'clust_coeff', 'norm_cc', 'deg_cent', 'norm_dc',
              'avg_neb_deg', 'norm_and', 'eccentry', 'norm_ecc', 'short_avg_len', 'norm_sal']
    # create rows with property data
    rows = np.array([page_rank, norm_pr, clust_coeff, norm_cc, deg_cent, norm_dc,
                     avg_neb_deg, norm_and, eccentry, norm_ecc, short_avg_len, norm_sal]).T
    # write to csv file
    with open(filename, 'w', newline='') as outfile:
        write = csv.writer(outfile)
        write.writerow(fields)
        write.writerows(rows)

# %% generate barabasi-albert graph and save properties as csv
# create directory to store graph properties and adjacency matrices
dir_prop = 'ba_graph_prop/'
dir_adjc = 'ba_graph_adjc/'
# clear all previous csv files from directory
clear_csv(dir_prop)
clear_csv(dir_adjc)

pcase = 0  # counter for model generation parameter
while icase != ncases:
    # BA graph generation
    G = nx.barabasi_albert_graph(n, m_BA[pcase])
    # save adjacency matrix to a csv file
    filename = dir_adjc + 'case_' + '{:04d}'.format(icase) + '.csv'
    np.savetxt(filename, nx.adjacency_matrix(G).todense(), delimiter=',')
    pcase += 1
    # check conditions
    if nx.is_connected(G):
        for istore in range(len(store_graphs)):
            if nx.is_isomorphic(store_graphs[istore], G):
                print(f'Isomorphic graphs found! Not a valid case.')
                continue
    else:
        print(f'Disconnected graph found! Not a valid case.')
        continue
    # if all condition satisfies then store graph and properties
    print(f'Valid case found! Running case {icase}: m = {m_BA[pcase]}')
    store_graphs[icase] = G
    icase += 1

    # page rank
    page_rank = list(nx.pagerank(G).values())
    norm_pr = normArr(page_rank)
    # clustering coefficient of each node
    clust_coeff = list(nx.clustering(G).values())
    norm_cc = normArr(clust_coeff)
    # degree centrality
    deg_cent = list(nx.degree_centrality(G).values())
    norm_dc = normArr(deg_cent)
    # average neighbor degree
    avg_neb_deg = list(nx.average_neighbor_degree(G).values())
    norm_and = normArr(avg_neb_deg)
    # eccentricity
    eccentry = list(nx.eccentricity(G).values())
    norm_ecc = eccentry / np.linalg.norm(eccentry)
    # shortest path length
    short_len = dict(nx.shortest_path_length(G))
    short_avg_len = [np.mean(list(short_len[i].values())) for i in range(n)]
    norm_sal = normArr(short_avg_len)

    # save each graph properties to a csv file
    filename = dir_prop + 'case_' + '{:04d}'.format(icase) + '.csv'
    # create field names for each property
    fields = ['page_rank', 'norm_pr', 'clust_coeff', 'norm_cc', 'deg_cent', 'norm_dc',
              'avg_neb_deg', 'norm_and', 'eccentry', 'norm_ecc', 'short_avg_len', 'norm_sal']
    # create rows with property data
    rows = np.array([page_rank, norm_pr, clust_coeff, norm_cc, deg_cent, norm_dc,
                    avg_neb_deg, norm_and, eccentry, norm_ecc, short_avg_len, norm_sal]).T
    # write to csv file
    with open(filename, 'w', newline='') as outfile:
        write = csv.writer(outfile)
        write.writerow(fields)
        write.writerows(rows)


# %% graph properties that are NP-Hard to compute
# generate a small erdos-renyi graph and save properties as csv
# create directory to store graph properties and adjacency matrices
dir_prop = 'er_graph_prop/'
dir_adjc = 'er_graph_adjc/'
# clear all previous csv files from directory
clear_csv(dir_prop)
clear_csv(dir_adjc)
# iterate over all requested p_ER values
icase = 0  # count valid cases to store
pcase = 0  # counter for model generation parameter
# start timer
start_main = time.perf_counter()
while icase != ncases / 2:
    # ER graph generation
    G = nx.erdos_renyi_graph(n, p_ER[pcase])
    # save adjacency matrix to a csv file
    filename = dir_adjc + 'case_' + '{:04d}'.format(icase) + '.csv'
    np.savetxt(filename, nx.adjacency_matrix(G).todense(), delimiter=',')
    pcase += 1
    # check conditions
    if nx.is_connected(G):
        if icase != 0:
            for istore in range(len(store_graphs)):
                if nx.is_isomorphic(store_graphs[istore], G):
                    print(f'Isomorphic graphs found! Not a valid case.')
                    continue
        else:
            # if running first case then store graph and properties
            store_graphs = {icase: G}
    else:
        print(f'Disconnected graph found! Not a valid case.')
        continue
    # if all condition satisfies then store graph and properties
    print(f'Valid case found! Running case {icase}: p = {p_ER[pcase]}')
    store_graphs[icase] = G
    icase += 1

    # calculate average path length
    avg_path_len, num_paths = simple_path_props(G)
    # compute normalized average path length
    norm_apl = normArr(avg_path_len)
    # compute normalized number of paths
    norm_np = normArr(num_paths)

    # save each graph properties to a csv file
    filename = dir_prop + 'case_' + '{:04d}'.format(icase) + '.csv'
    # create field names for each property
    fields = ['avg_path_len', 'norm_apl', 'num_paths', 'norm_np']
    # create rows with property data
    rows = np.array([avg_path_len, norm_apl, num_paths, norm_np]).T
    # write to csv file
    with open(filename, 'w', newline='') as outfile:
        write = csv.writer(outfile)
        write.writerow(fields)
        write.writerows(rows)
# end timer
end_main = time.perf_counter()
# time taken to generate graphs and compute properties
print(f'Time taken to compute all simple path properties: {end_main-start_main} seconds')

#%%
import timeit
a = []
m = [100, 500, 1000, 1500, 2000]
for i in range(5):
    G = nx.gnm_random_graph(n, m[i])
    # a.append(timeit.timeit(lambda: list(nx.eccentricity(G).values()), number=5000))
    start_main = time.perf_counter()
    for j in range(5000):
        short_len = dict(nx.shortest_path_length(G))
        short_avg_len = [np.mean(list(short_len[i].values())) for i in range(n)]
    end_main = time.perf_counter()
    a.append(end_main-start_main)
    print(a[-1])
print(f'Average time: {np.mean(a)}')



# # paths gives efficiency of pairs of nodes i and j where i < j
# start = time.perf_counter()
# # a = nx.approximation.traveling_salesman_problem(G)
# # a = nx.current_flow_betweenness_centrality(G)
# for j in range(1, n):
#     # print index of nodes
#     print(f'Computing simple paths between the nodes 0 and {j}')
#     paths = nx.all_simple_paths(G, source=0, target=j)
#     print(f'Number of simple paths between the nodes 0 and {j}: {len(paths)}')
# end = time.perf_counter()
# # time taken to compute efficiency
# print(f'Time taken to compute efficiency: {end-start} seconds')


# clustering(G),degree_centrality(G),average_neighbor_degree(G),eccentricity(G), shortest_path_length(G)
