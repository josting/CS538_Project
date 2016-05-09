
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


def draw_wtd_graph(G,title,clusters=None,showfig=True):
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
    plt.suptitle(title)
    if showfig:
        plt.show() # display
    else:
        plt.savefig("{}.png".format(title)) # save as png
        plt.close()

def networkx_mcl(G, expand_factor = 2, inflate_factor = 2, max_loop = 10 , mult_factor = 1):
    keys = G.node.keys()
    A = nx.adjacency_matrix(G, keys)
    M,clusters = mcl(np.array(A.todense()), expand_factor, inflate_factor, max_loop, mult_factor)
    for cluster_id in sorted(clusters):
        nodes = clusters[cluster_id]
        nodes = [keys[n] for n in nodes]
        clusters[cluster_id] = nodes    
    return M,clusters

def draw_wifi_graph(user):    
    new_records = load_wifi_records(user)
    all_wifi_records.extend(new_records)
    G = create_graph(new_records)
    M,clusters = networkx_mcl(G, expand_factor = 3, inflate_factor = 2, mult_factor = 2,max_loop = 60)
    for cluster,macs in clusters.items():
        for mac in macs:
            G.node[mac]["cluster"] = cluster
    draw_wtd_graph(G,user,clusters,0)    
    
def draw_wifi_graphs(users):
    all_wifi_records = []
    for user in users:
        new_records = load_wifi_records(user)
        all_wifi_records.extend(new_records)
        G = create_graph(new_records)
        M,clusters = networkx_mcl(G, expand_factor = 3, inflate_factor = 2, mult_factor = 2,max_loop = 60)
        for cluster,macs in clusters.items():
            for mac in macs:
                G.node[mac]["cluster"] = cluster
        draw_wtd_graph(G,user,clusters,0)
    
    # All users
    G = create_graph(all_wifi_records)
    M,clusters = networkx_mcl(G, expand_factor = 3, inflate_factor = 2, mult_factor = 2,max_loop = 60)
    for cluster,macs in clusters.items():
        for mac in macs:
            G.node[mac]["cluster"] = cluster
    draw_wtd_graph(G,"AllUsers",clusters,0)

def create_bt_graph(pairs):
    edges = [(mac1, mac2, 1) for (mac1,mac2) in pairs]
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    return G
    
def draw_bt_graph(user):
    #load the bt graph
    bt_traces = {}
    for user in users:
        for ts,pair in load_bt_records2(user):
            if ts not in bt_traces:
                bt_traces[ts] = []
            bt_traces[ts].append(pair)
    edges = {}
    trace_timestamps = sorted(bt_traces)
    for ts in trace_timestamps:
        f = lambda x: (ts - alpha) <= x and x <= (ts + alpha)
        x = map(bt_traces.__getitem__,  filter(f, trace_timestamps))
            
    all_wifi_records.extend(new_records)
    G = create_graph(new_records)
    M,clusters = networkx_mcl(G, expand_factor = 3, inflate_factor = 2, mult_factor = 2,max_loop = 60)
    for cluster,macs in clusters.items():
        for mac in macs:
            G.node[mac]["cluster"] = cluster
    draw_wtd_graph(G,user,clusters,0)
    
    
if __name__ == "__main__":
    users = []
    for fn in os.listdir(os.path.join(DATA_DIR, "uim_exp1_release")):
        path = os.path.join(DATA_DIR, "uim_exp1_release", fn)
        if os.path.isdir(path) and fn.startswith("User"):
            users.append(fn)
    #
    #load the bt graph
    bt_traces = {}
    for user in users:
        for ts,pair in load_bt_records2(user):
            if ts not in bt_traces:
                bt_traces[ts] = []
            bt_traces[ts].append(pair)
    edges = {}
    # idx = random.sample(xrange(len(timestamps)), 1500)
    trace_timestamps = sorted(bt_traces)
    for ts in trace_timestamps:
        f = lambda x: (ts - ALPHA) <= x and x <= (ts + ALPHA)
        pairs = reduce(list.__add__,reduce(list.__add__,map(bt_traces.__getitem__,  filter(f, trace_timestamps))))
        edges[ts] = pairs
    graph_orders = [(ts,len(pairs)) for ts,pairs in edges.items()]
    data = map(lambda x: x[1], graph_order)
    # stats
    mean = statistics.mean(data)
    stdev = statistics.stdev(data, xbar=mean)    
    max_order = max(data)
    f = lambda x: (mean - stdev) <= x[1] and x[1] <= (mean + stdev)
    samples = filter(f, graph_orders)
    for idx in random.sample(xrange(len(samples)), 5):
        ts = samples[idx][0]
        pairs = edges[ts]
        G = create_bt_graph(pairs)
        M,clusters = networkx_mcl(G, expand_factor = 3, inflate_factor = 2, mult_factor = 2,max_loop = 60)
        for cluster,macs in clusters.items():
            for mac in macs:
                G.node[mac]["cluster"] = cluster
        title = "UIM BT Traces at %s" % ts
        draw_wtd_graph(G, title,clusters, 1)
    raise
    draw_bt_graph(users[0])