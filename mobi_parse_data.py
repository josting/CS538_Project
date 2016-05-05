
import datetime as dt
import random
# import matplotlib as mpl
# import matplotlib.pyplot as plt

START_DT = dt.datetime(2009, 8, 17, 8) # 17/08/2009 08:00

activity = {}
with open(r"sigcomm2009\activity.csv") as activity_fd:
    for line in activity_fd.readlines():
        line = line.strip()
        if "#" in line:
            line = line[:line.index("#")]
        if not line:
            continue
        user_id, start_ts, end_ts = line.split(';')
        if user_id not in activity:
            activity[user_id] = []
        activity[user_id].append( (int(start_ts), int(end_ts)) )

def is_awake(user_id, ts, activity):
    for start_ts, end_ts in activity.get(user_id, []):
        if ts >= start_ts and ts <= end_ts:
            return True
    return False

transmission = {}
with open(r"sigcomm2009\transmission.csv") as transmission_fd:
    for line in transmission_fd.readlines():
        line = line.strip()
        if "#" in line:
            line = line[:line.index("#")]
        if not line:
            continue
        msg_type, msg_id, bytes, src_user_id, dst_user_id, ts, status = line.split(';')
        #if status != '0':
        #    continue
        if src_user_id not in transmission:
            transmission[src_user_id] = {}
        if dst_user_id not in transmission[src_user_id]:
            transmission[src_user_id][dst_user_id] = []
        ts = int(ts)
        transmission[src_user_id][dst_user_id].append(ts)
    
reception = {}
with open(r"sigcomm2009\reception.csv") as reception_fd:
    for line in reception_fd.readlines():
        line = line.strip()
        if "#" in line:
            line = line[:line.index("#")]
        if not line:
            continue
        msg_type, msg_id, src_user_id, dst_user_id, ts = line.split(';')
        if src_user_id not in reception:
            reception[src_user_id] = {}
        if dst_user_id not in reception[src_user_id]:
            reception[src_user_id][dst_user_id] = []
        ts = int(ts)
        reception[src_user_id][dst_user_id].append(ts)

drift_dict = {}
for src_user_id in sorted(reception):
    for dst_user_id in sorted(reception[src_user_id]):
        for rcp_ts in reception[src_user_id][dst_user_id]:
            if src_user_id not in transmission:
                continue
            transmissions = transmission[src_user_id].get(dst_user_id, None)
            if transmissions is None:
                continue
            if (src_user_id, dst_user_id) not in drift_dict:
                drift_dict[(src_user_id, dst_user_id)] = []
            diff = [abs(rcp_ts - trn_ts) for trn_ts in transmissions]
            idx = diff.index(min(diff))
            trn_ts = transmission[src_user_id][dst_user_id][idx]
            drift = trn_ts - rcp_ts
            drift_dict[(src_user_id, dst_user_id)].append((trn_ts, drift))

for (src_user_id, dst_user_id) in sorted(drift_dict):
    print src_user_id, dst_user_id, drift_dict[(src_user_id, dst_user_id)]
    break

            
proximity = {}
with open(r"sigcomm2009\proximity.csv") as proximity_fd:
    for line in proximity_fd.readlines():
        line = line.strip()
        if "#" in line:
            line = line[:line.index("#")]
        if not line:
            continue
        ts, user_id, seen_user_id, major_code, minor_code = line.split(';')
        ts = int(ts)
        if ts not in proximity:
            proximity[ts] = []
        proximity[ts].append((user_id, seen_user_id))

    
def visit(node, edges, unvisited):
    if node not in unvisited:
        return []
    unvisited.remove(node)
    my_network = [node]
    for (node1, node2) in edges:
        if node == node1 and node2 in unvisited:
            my_network.extend(visit(node2, edges, unvisited))
        elif node == node2 and node1 in unvisited:
            my_network.extend(visit(node1, edges, unvisited))
    return my_network

def get_networks(nodes, edges):
    networks = []
    unvisited = list(nodes)
    while unvisited:
        node = unvisited[0]
        my_network = []
        networks.append(visit(node, edges, unvisited))
    return map(sorted,(map(set,networks)))

MAX_RNG = 75
timestamps = sorted(proximity)

#write traces to user.dat files
user_fds = {}
for ts in timestamps:
    for (user_id, seen_id) in proximity[ts]:
        if user_id not in user_fds:
            fd = open(r"mobiclique\%s.dat" % user_id, 'w')
            last_ts = -1
            user_fds[user_id] = [fd, last_ts]
        else:
            [fd, last_ts] = user_fds[user_id]
        if last_ts != ts:
            if last_ts > 0:
                fd.write('\n')
            fd.write("{} {} {}".format(ts, user_id, seen_id))
        else:
            fd.write(",{}".format(seen_id))
        user_fds[user_id][1] = ts

for (fd, last_ts) in user_fds.values():
    fd.close()
        

#
networks = []
n_networks = []
max_size = []

idx = random.sample(xrange(len(timestamps)), 1500)
idx.sort()
sample_timestamps = map(timestamps.__getitem__, idx)
sample_dts = map(lambda ts: START_DT + dt.timedelta(seconds=ts),sample_timestamps)
for ts in sample_timestamps:
    other_timestamps = filter(lambda x: abs(x-ts) < MAX_RNG, timestamps)
    edges = sorted(set(reduce(list.__add__, [proximity[x] for x in other_timestamps])))
    nodes = sorted(set(reduce(list.__add__, map(list, edges))))
    new_networks = get_networks(nodes, edges)
    networks.append(new_networks)
    n_networks.append(len(new_networks))
    max_size.append(max(map(len,new_networks)))

fd = open("output2.csv", 'w')
for vals in zip(sample_dts, n_networks, max_size):
    fd.write(','.join(map(str,(vals))))
    fd.write('\n')

fd.close()
