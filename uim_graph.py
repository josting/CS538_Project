
import os
import datetime as dt
import networkx as nx
from mcl.mcl_clustering import mcl
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random

from const import *
from uim_location import *

def create_graph(records):
    pairs = {}
    counts = {}
    pair = namedtuple('pair', ['j','k'])
    all_record_pairs = []
    for t, macs in records:
        record_pairs = []
        for (j, mac1) in enumerate(macs):
            if mac1 not in counts:
                counts[mac1] = 0
            counts[mac1] += 1
            for (k, mac2) in enumerate(macs[j+1:]):
                p = pair(mac1, mac2)
                if p not in pairs:
                    pairs[p] = 0
                pairs[p] += 1
                record_pairs.append(p)
        all_record_pairs.append(record_pairs)
    support = {}
    for p, c in pairs.items():
        support[p] = float(c) / float(min(counts[p.j], counts[p.k]))
    edges = [(p.j, p.k, s) for (p,s) in support.items()]
    unwtd = [(p.j, p.k) for p in support]
    G = nx.Graph(unwtd)
    G.add_weighted_edges_from(edges)
    return G


def create_edges(records):
    edges = []
    all_macs = []
    for t, macs in records:
        for (i, mac1) in enumerate(macs):
            for mac2 in macs[i+1:]:
                edges.append((mac1,mac2))
    return edges
        
    # pairs = {}
    # counts = {}
    # pair = namedtuple('pair', ['j','k'])
    # all_record_pairs = []
    # for t, macs in records:
        # record_pairs = []
        # for j in range(len(macs)):
            # if macs[j] not in counts:
                # counts[macs[j]] = 0
            # counts[macs[j]] += 1
            # for k in range(j+1,len(macs)):
                # p = pair(macs[j], macs[k])
                # if p not in pairs:
                    # pairs[p] = 0
                # pairs[p] += 1
                # record_pairs.append(p)
        # all_record_pairs.append(record_pairs)
    # support = {}
    # for p, c in pairs.items():
        # support[p] = float(c) / float(min(counts[p.j], counts[p.k]))

    # ratios = []
    # for i in range(len(all_record_pairs)):
        # record_pairs = all_record_pairs[i]
        # data = [support[p] for p in record_pairs]
        # if len(data) < 2:
            # ratios.append((0, records[i]))
        # else:
            # mean = statistics.mean(data)
            # stdev = statistics.stdev(data, xbar=mean)
            # ratios.append((stdev / mean, records[i]))

    # ratios = sorted(ratios, key=lambda x:x[0])

    # good_records = []
    # all_macs = set()
    # for _, record in ratios:
        # size = len(all_macs)
        # all_macs |= set(record[1])
        # '''Only add records that provide new information '''
        # if len(all_macs) > size:
            # good_records.append(record)

    # return good_records, all_macs

def draw_wtd_graph(G,clusters=None,showfig=True):
    ''' Based on an example by Aric Hagberg (hagberg@lanl.gov)'''
    
    pos=nx.spring_layout(G) # positions for all nodes

    # nodes
    if clusters is None:
        nx.draw_networkx_nodes(G,pos,node_size=50)
    else:
        colors = [(random.random(),random.random(),random.random()) for i in xrange(255)]
        new_map = mpl.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256)
        for (i,(cluster,macs)) in enumerate(clusters.items()):
            nx.draw_networkx_nodes(G,pos,nodelist=macs,node_size=50,node_color=new_map(i))#,cmap=new_map)
    #
    elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >0.5]
    esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <=0.5]
    # edges
    nx.draw_networkx_edges(G,pos,edgelist=elarge, width=.75)
    nx.draw_networkx_edges(G,pos,edgelist=esmall, width=.5,alpha=0.5,edge_color='k',style='dashed')

    # labels
    # nx.draw_networkx_labels(G,pos,font_size=20,font_family='sans-serif')

    plt.axis('off')
    if showfig:
        plt.show() # display
    else:
        plt.savefig("weighted_graph.png") # save as png

def networkx_mcl(G, expand_factor = 2, inflate_factor = 2, max_loop = 10 , mult_factor = 1):
    keys = G.node.keys()
    A = nx.adjacency_matrix(G, keys)
    M,clusters = mcl(np.array(A.todense()), expand_factor, inflate_factor, max_loop, mult_factor)
    for cluster_id in sorted(clusters):
        nodes = clusters[cluster_id]
        nodes = [keys[n] for n in nodes]
        clusters[cluster_id] = nodes    
    return M,clusters
    
    
if __name__ == "__main__":
    users = []
    for fn in os.listdir(os.path.join(DATA_DIR, "uim_exp1_release")):
        path = os.path.join(DATA_DIR, "uim_exp1_release", fn)
        if os.path.isdir(path) and fn.startswith("User"):
            users.append(fn)
    #
    for user in users:
        all_wifi_records = load_wifi_records(user)
        G = create_graph(all_wifi_records)
        M,clusters = networkx_mcl(G, expand_factor = 2, inflate_factor = 2, mult_factor = 2,max_loop = 60)
        for cluster,macs in clusters.items():
            for mac in macs:
                G.node[mac]["cluster"] = cluster

        draw_wtd_graph(G,clusters)
        
        break
    

    