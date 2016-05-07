#!/usr/bin/python

import os
import glob
import statistics
import numpy as np
import networkx as nx
from mcl_clustering import networkx_mcl
from os import path
from datetime import datetime, timedelta
from collections import namedtuple

DATA_DIR = os.getenv('DATA_DIR', r"C:\Users\Jon\Documents\UIUC\CS 538\project\data")

def load_records_from_file(filepath):
    records = []
    t = None
    macs = []
    with open(filepath) as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            if l.startswith('t='):
                if len(macs) > 0:
                    records.append((t, macs))
                t = datetime.strptime(l[2:], '%m-%d-%Y %H:%M:%S')
                macs = []
            else:
                macs.append(l)
    return records


def load_wifi_records(user):
    pattern = path.join(DATA_DIR, "uim_exp1_release", user, "btlog", "w*")
    files = glob.glob(pattern)
    records = []
    for f in files:
        records.extend(load_records_from_file(f))
    return records

def load_bt_records(user):
    pattern = path.join(DATA_DIR, "uim_exp1_release", user, "btlog", "s*")
    files = glob.glob(pattern)
    records = []
    for f in files:
        records.extend(load_records_from_file(f))
    return records


def good_wifi_set(records):
    pairs = {}
    counts = {}
    pair = namedtuple('pair', ['j','k'])
    all_record_pairs = []
    for t, macs in records:
        record_pairs = []
        for j in range(len(macs)):
            if macs[j] not in counts:
                counts[macs[j]] = 0
            counts[macs[j]] += 1
            for k in range(j+1,len(macs)):
                p = pair(macs[j], macs[k])
                if p not in pairs:
                    pairs[p] = 0
                pairs[p] += 1
                record_pairs.append(p)
        all_record_pairs.append(record_pairs)
    support = {}
    for p, c in pairs.items():
        support[p] = float(c) / float(min(counts[p.j], counts[p.k]))

    ratios = []
    for i in range(len(all_record_pairs)):
        record_pairs = all_record_pairs[i]
        data = [support[p] for p in record_pairs]
        if len(data) < 2:
            ratios.append((0, records[i]))
        else:
            mean = statistics.mean(data)
            stdev = statistics.stdev(data, xbar=mean)
            ratios.append((stdev / mean, records[i]))

    ratios = sorted(ratios, key=lambda x:x[0])

    good_records = []
    all_macs = set()
    for _, record in ratios:
        size = len(all_macs)
        all_macs |= set(record[1])
        # Only add records that provide new information
        if len(all_macs) > size:
            good_records.append(record)

    return good_records, all_macs

def sim_graph(records, all_macs, sim_threshold):
    bits = {}
    macs = sorted(all_macs)
    for i in range(len(macs)):
        bits[macs[i]] = i

    vectors = []
    for record in records:
        vector = np.zeros(len(all_macs))
        for mac in record[1]:
            vector[bits[mac]] = 1
        vectors.append(vector)

    pair = namedtuple('pair', ['p', 'q'])
    tanimoto = {}
    for p in range(len(vectors)):
        for q in range(p+1,len(vectors)):
            pq = pair(p,q)
            p2 = np.dot(vectors[p], vectors[p])
            q2 = np.dot(vectors[q], vectors[q])
            dot = np.dot(vectors[p], vectors[q])
            tanimoto[pq] = dot / (p2 + q2 - dot)

    vertexes = [i for i in range(len(vectors))]
    edges = []
    for pq, t in tanimoto.items():
        if t > sim_threshold:
            edges.append((pq.p, pq.q))

    return vertexes, edges, vectors, bits

def mcl_cluster(vs, es):
    g = nx.Graph()
    for i in range(len(vs)):
        g.add_node(i)
    for n0, n1 in es:
        g.add_edge(n0, n1)

    M, clusters = networkx_mcl(g)
    locations = []
    for _, cluster in clusters.items():
        locations.append(set(cluster))
    return locations


def final_locations(locations, vectors):
    sigs = []
    l = len(vectors[0])
    for location in locations:
        sig = [0 for i in range(l)]
        for v in location:
            vec = vectors[v]
            for i in range(l):
                sig[i] |= int(vec[i])
        sigs.append(sig)

    marked = {}
    for i in range(len(sigs)):
        marked[i] = False

    # Combine locations that are a subset of another location
    final_locations = []
    final_sigs = []
    for i in range(len(sigs)):
        if marked[i]:
            continue
        location = locations[i]
        for j in range(i+1, len(sigs)):
            if marked[j]:
                continue

            match = True
            for k in range(len(sigs[i])):
                if sigs[i][k] == 0 and sigs[j][k] == 1:
                    match = False
                    break

            if match:
                location |= locations[j]
                marked[j] = True

        final_locations.append(location)
        final_sigs.append(sigs[i])
        marked[i] = True

    return final_locations, final_sigs

def classify_wifi_records(all_wifi_records, sigs, bits):
    vectors = []
    for record in all_wifi_records:
        vector = np.zeros(len(bits))
        for mac in record[1]:
            vector[bits[mac]] = 1
        vectors.append(vector)

    location_records = []
    for p in range(len(vectors)):
        max_sim = 0
        match_location = -1
        for q in range(len(sigs)):
            p2 = np.dot(vectors[p], vectors[p])
            q2 = np.dot(sigs[q], sigs[q])
            dot = np.dot(vectors[p], sigs[q])
            sim = dot / (p2 + q2 - dot)
            if sim > max_sim:
                match_location = q
                max_sim = sim
        location_records.append((all_wifi_records[p][0], all_wifi_records[p][1], match_location))

    return location_records

def map_bt_locations(all_bt_records, location_wifi_records, alpha):
    location_bt_records = []
    for t, _, loc in location_wifi_records:
        start = t - alpha
        stop = t + alpha
        for bt_t, macs in all_bt_records:
            if bt_t >= start and bt_t <= stop:
                location_bt_records.append((bt_t, macs, loc))
    return location_bt_records

def classify_bt_records(all_bt_records, location_bt_records, num_locs):
    all_bt_macs = set()
    for _, macs in all_bt_records:
        all_bt_macs |= set(macs)

    mu = len(all_bt_macs)
    count_locs = [0 for i in range(num_locs)]
    for _,_,loc in location_bt_records:
        count_locs[loc] += 1
    prob_locs = []
    m = float(len(all_bt_macs))
    for l in range(num_locs):
        prob_locs.append(count_locs[l] / m)

    pair = namedtuple('mac_loc', ['mac','loc'])
    count_macs = {}
    for _, macs, loc in location_bt_records:
        for m in macs:
            ml = pair(m,loc)
            if ml not in count_macs:
                count_macs[ml] = 0
            count_macs[ml] += 1

    def f_L(macs, l):
        f = 1.0
        for m in macs:
            ml = pair(m, l)
            cm = 0
            if ml in count_macs:
                cm = count_macs[ml]
            f *=((cm + 1) / (count_locs[l] + mu)) * prob_locs[l]
        return f

    # Determine minimum good value of `f`
    # using only the training location_bt_records
    min_f = 1.0
    for t, macs, _ in location_bt_records:
        for l in range(num_locs):
            f = f_L(macs, l)
            if f < min_f:
                min_f = f

    all_bt_location_records = []
    for t,macs in all_bt_records:
        max_loc = None
        max_f = 0
        for l in range(num_locs):
            f = f_L(macs, l)
            if f > max_f:
                max_loc = l
                max_f = f

        # if max_f is not large enough assign `unknown` location
        if max_f < min_f:
            max_loc = num_locs

        all_bt_location_records.append((t, macs, max_loc))

    return all_bt_location_records


def classify_user_data(user, sim_threshold, alpha):
    all_wifi_records = load_wifi_records(user)
    print('Loaded all wifi records %d' % len(all_wifi_records))
    good_records, all_macs = good_wifi_set(all_wifi_records)
    print('Found %d good wifi records' % len(good_records))
    vs, es, vectors, bits = sim_graph(good_records, all_macs, sim_threshold)
    print('Computed similarity graph')
    candidate_locations = mcl_cluster(vs, es)
    print('Clustered graph into locations')
    locations, sigs = final_locations(candidate_locations, vectors)
    print('Found %d locations' % len(locations))
    location_wifi_records = classify_wifi_records(all_wifi_records, sigs, bits)
    print('Classified all wifi records into locations')

    all_bt_records = load_bt_records(user)
    print('Loaded all bt records')
    location_bt_records = map_bt_locations(all_bt_records, location_wifi_records, alpha)
    print('Mapped bt records to wifi locations')
    all_bt_location_records = classify_bt_records(all_bt_records, location_bt_records, len(locations))
    print('Classified all bt records into locations')

    return all_bt_location_records

if __name__ == '__main__':
    sim_threshold = 0.1
    alpha = timedelta(seconds=60)



    users = os.listdir(path.join(DATA_DIR, "uim_exp1_release"))
    for user in users:
        print('User: %s' % user)
        if not path.isdir(path.join(DATA_DIR, "uim_exp1_release", user)):
            continue
        filepath = path.join(DATA_DIR, "uim_mcl", user)
        if path.exists(filepath):
            continue

        records = classify_user_data(user, sim_threshold, alpha)
        with open(filepath, 'w') as f:
            for t,macs,loc in records:
                f.write('%s %s %s\n' % (t.isoformat(), loc, ','.join(macs)))


