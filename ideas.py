#!/usr/bin/python

import os
import glob
import numpy as np
import networkx as nx

from mcl_clustering import networkx_mcl
from os import path
from datetime import datetime, timedelta
from collections import namedtuple

DATA_DIR = os.getenv('DATA_DIR', r"C:\Users\Jon\Documents\UIUC\CS 538\project\data")

def load_loc_records_from_file(filepath):
    records = []
    with open(filepath) as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            t, loc, macs = l.split()
            macs = set(macs.split(','))
            records.append((datetime.strptime(t, '%Y-%m-%dT%H:%M:%S'), loc, macs))

    return records

def load_loc_records(user):
    filepath = path.join(DATA_DIR, "uim_classified", user)
    return load_loc_records_from_file(filepath)

# 2 hour buckets
def time_of_day(t):
    return t.time().hour / 2

def weekday_type(t):
    wd = t.date().weekday()
    if wd >= 5:
        return 'weekend'
    else:
        return 'weekday'

def load_graph(filepath):
    g = nx.Graph()
    with open(filepath, 'r') as f:
        nodes_loaded = False
        for l in f:
            l = l.strip()
            if not l:
                continue
            if not nodes_loaded:
                nodes_loaded = True
                nodes = l.split()
                for n in nodes:
                    g.add_node(int(n))
            else:
                e0, e1, w = l.split()
                e0, e1, w = int(e0), int(e1), int(w)
                g.add_edge(e0, e1, weight=w)
    return g

def save_graph(g, filepath):
    with open(filepath, 'w') as f:
        for node in g.nodes():
            f.write('%d ' % node)

        f.write('\n')

        for e0, e1, data in g.edges(data=True):
            f.write('%d %d %d\n' % (e0 ,e1, data['weight']))

def create_raw_graph(user, records, load=True):
    filepath = user + '_raw.nxg'
    if load:
        try:
            g = load_graph(filepath)
            print('Loaded graph')
            return g
        except:
            pass

    print('Constructing graph')
    g = nx.Graph()
    #for i in range(len(records)):
    #    g.add_node(i)

    for i in range(len(records)):
        t0, _, macs0 = records[i]
        td0 = time_of_day(t0)
        wd0  = weekday_type(t0)
        for j in range(i+1, len(records)):
            t1, _, macs1  = records[j]
            td1 = time_of_day(t1)
            wd1  = weekday_type(t1)
            w = len(macs0 & macs1)
            if td0 == td1 and wd0 == wd1 and w > 1:
                print('Adding edge', i, j, w)
                g.add_edge(i,j, weight=w)


    save_graph(g, filepath)
    print('Created graph')

    return g

def create_loc_graph(user, records, load=True):
    filepath = user + '_loc.nxg'
    if load:
        try:
            g = load_graph(filepath)
            print('Loaded graph')
            return g
        except:
            pass

    print('Constructing graph')
    g = nx.Graph()
    #for i in range(len(records)):
    #    g.add_node(i)

    for i in range(len(records)):
        t0, loc0, macs0 = records[i]
        td0 = time_of_day(t0)
        wd0  = weekday_type(t0)
        for j in range(i+1, len(records)):
            t1, loc1, macs1  = records[j]
            td1 = time_of_day(t1)
            wd1  = weekday_type(t1)
            w = len(macs0 & macs1)
            if loc0 == loc1 and td0 == td1 and wd0 == wd1 and w > 1:
                print('Adding edge', i, j, w)
                g.add_edge(i,j, weight=w)


    save_graph(g, filepath)
    print('Created graph')

    return g

def star_cluster(g):
    vs = g.nodes()
    es = g.edges()
    degrees = []
    for v in vs:
        degree = 0
        for e in es:
            if v in e:
                degree += 1
        degrees.append((v, degree))

    degrees = sorted(degrees, key=lambda x:x[1], reverse=True)
    marked = {}
    for v in vs:
        marked[v] = False

    locations = []
    for v,d in degrees:
        if marked[v]:
            continue
        location = set()
        location.add(v)
        for e in es:
            if e[0] == v:
                if not marked[e[1]]:
                    location.add(e[1])
                    marked[e[1]] = True
            if e[1] == v:
                if not marked[e[0]]:
                    location.add(e[0])
                    marked[e[0]] = True
        locations.append(location)

    return locations



if __name__ == '__main__':
    user = 'User15'
    records = load_loc_records(user)
    print('Loaded %d records' % len(records))

    raw_g = create_raw_graph(user, records, False)
    loc_g = create_loc_graph(user, records, False)

    raw_M, raw_clusters = networkx_mcl(
            raw_g,
        )
    print("Raw clusters", len(raw_clusters))
    print("Clusters:")
    t = 0 
    for k, v in raw_clusters.items():
        t += len(v)
        print(k, len(v))
    print(t)



