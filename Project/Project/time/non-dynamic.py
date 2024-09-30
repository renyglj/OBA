import heapq
import itertools
from itertools import product

import numpy as np
from tqdm import tqdm

from option import args_parser
from partition_point import initialize, partition_point


def all_point_sequence(args, p):
    # p = partition_point(args)  #备选分割点，一个client有4个点
    #对10个clients的4个分割点进行全排列得到4**10个分割点序列
    seq = itertools.product(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9])
    seq = list(seq)
    return seq
def initialize_bandwidth(args, p):  #对每一个分割点序列，中间层数据量之后，初始按数据量大小分配带宽
    seqs = all_point_sequence(args, p)
    clients, server = initialize(args)
    mid = [0] * args.num_client
    sum_mid = [0] * len(seqs)
    for j, seq in enumerate(seqs):
        seq = list(seq)
        for i, r in enumerate(seq):
            mid[i] = server.mid_data[r]
            sum_mid[j] += mid[i]
    return sum_mid


def initialize_sequence(args, seqs, sum_mid):   #初始情况：j=1时，10个client同时开始，4**10个分割点序列选1个
    # seqs = all_point_sequence(args)  #4**10个子列表，每个子列表有10个点对应10个client
    B = args.bandwidth
    clients, server = initialize(args)
    # sum_mid = initialize_bandwidth(args)
    beta = [[0]*args.num_client for i in range(len(seqs))]
    t_1, t_2, = [[0]*args.num_client for i in range(len(seqs))], [[0]*args.num_client for i in range(len(seqs))]
    for k, seq in enumerate(seqs):
        for i, r in enumerate(seq):  #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
            beta[k][i] = server.mid_data[r]/sum_mid[k]
            t_1[k][i] = clients[i].cfw_time[0][r]
            t_2[k][i] = t_1[k][i] + (server.mid_data[r]/(B*beta[k][i]))*1000
            # t_3[k][i] = t_2[k][i] + server.sfw_time[r+1] + server.mid_data[r]/(B*beta[i]) + t_w +clients[i].cbw_time[0][r]
            # t_4[k][i] = t_3[k][i] + args.para[r]/(B*beta[i])
    count = [0] * len(seqs)
    for k in range(len(seqs)):
        temp = (t_1[k][0]+t_2[k][0])/2
        count[k] = 0
        for i in range(1, args.num_client):
            if t_1[k][i] <= temp:
                count[k] += 1
    best_seq = min(count)
    k = count.index(best_seq)
    # print(k)
    return k, seqs[k], beta[k]

def wait(args, seq, k, sum_mid, c, beta, j):
    clients, server = initialize(args)
    t_1, t_2 = [0]*args.num_client, [0]*args.num_client
    B = args.bandwidth
    for i, r in enumerate(seq):
        t_1[i] = clients[i].cfw_time[0][r]
        t_2[i] = t_1[i] + server.mid_data[r] / (B * beta[i])
        clients[i].clock = t_2[i]    #clients的任务到达服务器的时间
        # t_3[i] = t_2[i] + server.sfw_time[r + 1] + server.mid_data[r] / (B * beta[i]) + t_w
        # t_4[i] = t_3[i] + args.para[r] / (B * beta[i])
    arrive_time = sorted(t_2)
    first = t_2.index(arrive_time[0])   #第一个到达的client
    point = seq[first]   #第一个到达client的分割点
    # print(point)
    # print('first:', first)
    # print('second:', t_2.index(arrive_time[2]))
    # print('arrive time:', arrive_time)
    # print("t_2", t_2)
    # print("c", c)
    # print('comp:', server.sfw_time[point])
    # print(':', arrive_time[0]+server.sfw_time[point]-arrive_time[1])
    earlier_c = 0
    c_arrivetime = 0
    for i1, t in enumerate(arrive_time):    #client c 的到达时间
        if t_2.index(arrive_time[i1]) == c:
            c_arrivetime = arrive_time[i1]
    # print('c_arrive:', c_arrivetime)
    for i2, t in enumerate(arrive_time):    #在c之前到达的client数量
        if arrive_time[i2] <= c_arrivetime:
            earlier_c += 1
    # print('num_earlier:', earlier_c)
    wait_time = [[0]*2 for i in range(earlier_c)]
    for t in range(earlier_c):  #在c之前的client的等待时间
        wait_time[t][0] = t_2.index(arrive_time[t])
        wait_time[t][1] = arrive_time[t]
    # print("_________", earlier_c)
    wait_time[0][1] = 0
    s_comptime = [[0]*2 for i in range(earlier_c)]
    for t in range(earlier_c):
        s_comptime[t][0] = t_2.index(arrive_time[t])
        s_comptime[t][1] = arrive_time[t]
    s_comptime[0][1] = arrive_time[0] + wait_time[0][1] + server.sfw_time[point]
    for t in range(1, earlier_c):
        wait_time[t][1] = max(s_comptime[t-1][1] - arrive_time[t], 0)
        s_comptime[t][1] = arrive_time[t] + wait_time[t][1] + server.sfw_time[t_2.index(arrive_time[t])]
    # print('wait_time:', wait_time)
    wait = wait_time[len(wait_time)-1][1]
    # print('wait', wait)
    return wait
def wait_time(args, current_c, previous_c, previous_p, arrive_t, wait_t):
    clients, server = initialize(args)
    if previous_c==None:
        w_t = 0
    else:
        w_t = max(arrive_t[previous_c][-2] + wait_t[previous_c][-1] + server.sfw_time[previous_p] - arrive_t[current_c][-1], 0)
    wait_t[current_c].append(w_t)
    print('wait_time:', wait_t)
    return wait_t[current_c][-1]

# def first_finish(args, seq, k, sum_mid, t_w, j):
#     clients, server = initialize(args)
#     t_1, t_2, t_3, t_4 = [0] * 10, [0] * 10, [0] * 10, [0] * 10
#     B = args.bandwidth
#     beta = [[0]*args.num_comm for i in range(args.num_client)]
#     for i, r in enumerate(seq):
#         beta[i] = server.mid_data[r] / sum_mid[k]
#         t_1[i] = clients[i].cfw_time[0][r]
#         t_2[i] = t_1[i] + server.mid_data[r] / (B * beta[i])
#     min_time = min(t_2)
#     last_c = t_2.index(min_time)
#     point = seq[last_c]
#     first_finish = min_time + t_w + server.sfw_time[point] + server.mid_data[point]/B + clients[last_c].cbw_time[0][point] + clients[last_c].para / (B * beta[last_c])
#     return first_finish
def first_finish_time(args, seq, k,p, beta, sum_mid, j, arrive_t, finish_t, previous_c, previous_p, wait_t):
    clients, server = initialize(args)
    sort_arr = sorted(arrive_t, key=(lambda x: x[-1]))
    min_time = sort_arr[0][-1]
    for i, l in enumerate(arrive_t):
        if min_time in l:
            index = (i, l.index(min_time))
            break
    last_c = list(index)[0]
    point = seq[last_c]
    f_t = min_time + wait_time(args, last_c, previous_c, previous_p, arrive_t, wait_t) + server.sfw_time[point] \
          + (server.mid_data[point] / args.bandwidth)*1000 + clients[last_c].cbw_time[0][point]
    finish_t[last_c].append(f_t)
    return finish_t, last_c, point
def finish_time(args, seq, k, beta, sum_mid, j, client, point, arrive_t, finish_t, prevoius_c, previous_p, wait_t):
    clients, server = initialize(args)
    print('****client,j', client, j)
    if (len(finish_t[client])-1) % 5 == 0:
        f_t = arrive_t[client][-1] \
              + wait_time(args, client, prevoius_c, previous_p, arrive_t, wait_t) + server.sfw_time[point] \
              + (server.mid_data[point] / args.bandwidth)*1000 + clients[client].cbw_time[0][point] \
              + (clients[client].para / (args.bandwidth * beta[client]))*1000 + (clients[client].para / args.bandwidth)*1000
        # finish_t[client].append(f_t)
    else:
        f_t = arrive_t[client][-1] \
              + wait_time(args, client, prevoius_c, previous_p, arrive_t, wait_t) + server.sfw_time[point] \
              + (server.mid_data[point] / args.bandwidth)*1000 + clients[client].cbw_time[0][point]
    finish_t[client].append(f_t)
    return finish_t

def finish_time1(args, seq, k, beta, sum_mid, j, client, point, arrive_t, finish_t, agg):
    clients, server = initialize(args)
    print('****client,j', client, j)
    if (len(finish_t[client])) % 5 == 0:
        agg = agg+1
        f_t = arrive_t[client][-1] \
              + wait(args, seq, k, sum_mid, client, beta, j) + server.sfw_time[point] \
              + (server.mid_data[point] / args.bandwidth)*1000 + clients[client].cbw_time[0][point] \
              + (clients[client].para[0][point] / (args.bandwidth * beta[client]))*1000 + (clients[client].para[0][point] / args.bandwidth)*1000
        # finish_t[client].append(f_t)
    else:
        f_t = arrive_t[client][-1] \
              + wait(args, seq, k, sum_mid, client, beta, j) + server.sfw_time[point] \
              + (server.mid_data[point] / args.bandwidth)*1000 + clients[client].cbw_time[0][point]
    finish_t[client].append(f_t)
    return finish_t

def first_arrive_time(args, seq, j, beta, arrive_t, finish_t):
    clients, server = initialize(args)
    t_1, t_2 = [0] * args.num_client, [0] * args.num_client
    for i, r in enumerate(seq):
        t_1[i] = finish_t[i][j-1] + clients[i].cfw_time[0][r]
        # print('1:', t_1[i])
        t_2[i] = t_1[i] + (server.mid_data[r] / (args.bandwidth * beta[i]))*1000
        # print('2:', t_2[i])
        arrive_t[i].append(t_2[i])
    return arrive_t

def arrive_time(args, seq, j, finish_t, beta, arrive_t, client, point):
    clients, server = initialize(args)
    t_1, t_2 = [0] * args.num_client, [0] * args.num_client
    for i, r in enumerate(seq):
        t_1[i] = finish_t[i][-1] + clients[i].cfw_time[0][r]
        t_2[i] = t_1[i] + (server.mid_data[r] / (args.bandwidth * beta[i]))*1000
        arrive_t[i].append(t_2[i])
    # t = finish_t[client][-1] +clients[client].cfw_time[0][point] + server.mid_data[point]/(args.bandwidth * beta[client])
    # arrive_t[client].append(t)
    return arrive_t

def find_index(c, arrive_t):
    for i, l in enumerate(arrive_t):
        if c in l:
            index = (i, l.index(c))
            break
    c_index = list(index)[0]
    return c_index

def optimize_sequence(args, seqs, k, p, beta, j, arrive_t, finish_t, temp_client, temp_point, sum_mid):
    clients, server = initialize(args)
    sort_arr_t = sorted(arrive_t, key=(lambda x: x[-1]))
    c = sort_arr_t[1][-1]
    for i, l in enumerate(arrive_t):
        if c in l:
            index = (i, l.index(c))
            break
    temp_c = list(index)[0]
    temp_p = seqs[k][temp_c]
    run = True
    min_count = 0
    count = [0] * len(seqs)
    overlap_c = [[] for i in range(len(seqs))]
    if sort_arr_t[0][-1] < finish_t[temp_c][-1] + clients[temp_c].cfw_time[0][temp_p]:
        # run = False
        run = True
    if run == False:
        temp_t = (finish_t[temp_c][-1] + clients[temp_c].cfw_time[0][temp_p] + finish_t[temp_c][-1] +
                  clients[temp_c].cfw_time[0][temp_p] + server.mid_data[temp_p] / (args.bandwidth * beta[temp_c])) / 2
        min_f_t = [0] * len(seqs)
        f_t = [[0] * args.num_client for i in range(len(seqs))]
        for k1, seq in enumerate(seqs):
            if seq[temp_c] >= temp_p:
                for i, r in enumerate(seq):
                    if finish_t[i][-1] + clients[i].cfw_time[0][r] < temp_t:
                        count[k1] += 1
                        overlap_c[k1].append(i)
                if count[k1] != 0:
                    beta = compute_beta(args, seqs, k1, overlap_c, beta, count[k1])
                    for i in overlap_c[k1]:
                        arr_t = finish_t[i][-1] + clients[i].cfw_time[0][seq[i]] + (server.mid_data[seq[i]] / (args.bandwidth * beta[i]))*1000
                        f_t[k1][i] = arr_t + wait(args, seq, k1, sum_mid, i, beta, j) + server.sfw_time[i] + (server.mid_data[i] / args.bandwidth)*1000 + clients[i].cbw_time[0][seq[i]]
                    min_f_t[k1] = min(filter(lambda x: x > 0, f_t[k1]))
        if min_f_t != 0:
            min_f_t_t = min(filter(lambda x: x > 0, min_f_t))
            opt_seq = min_f_t.index(min_f_t_t)
            temp_c = f_t[opt_seq].index(min_f_t_t)
            temp_p = seqs[opt_seq][temp_c]
        else:
            opt_seq = k
    else:
        opt_seq = k
    return opt_seq, temp_c, temp_p, count[k], overlap_c

def compute_beta(args, seqs, opt_seq, c, beta, count):
    clients, server = initialize(args)
    mid = [0]*args.num_client
    sum_mid = 0
    for i, r in enumerate(seqs[opt_seq]):
        if i in c[opt_seq]:
            mid[i] = server.mid_data[r]
            sum_mid += mid[i]
    for i, r in enumerate(seqs[opt_seq]):
        if i in c[opt_seq]:
            beta[i] = server.mid_data[r]/sum_mid
    # for i in c[opt_seq]:
    #     beta[i] = 1/count
    return beta

def point_main(args):
       p = partition_point(args)
       # p = [[21] * 4 for i in range(args.num_client)]
       print(p)
       seqs = all_point_sequence(args, p)
       sum_mid_data = initialize_bandwidth(args, p)
       k, seq, beta = initialize_sequence(args, seqs, sum_mid_data)
       arrive_t = [[0]*1 for i in range(args.num_client)]
       finish_t = [[0]*1for i in range(args.num_client)]
       wait_t = [[] for i in range(args.num_client)]
       for i in range(args.num_client):
           arrive_t[i][0] = 0
           finish_t[i][0] = 0
       previous_c = None
       previous_p = None
       agg = 0
       arrive_t = first_arrive_time(args, seq, 1, beta, arrive_t, finish_t)
       finish_t, temp_client, temp_point = first_finish_time(args, seq, k, p, beta, sum_mid_data, 1, arrive_t, finish_t, previous_c, previous_p, wait_t)
       # filename = 'write_data.txt'
       for j in range(1, args.num_comm+1):
           print('k, c', k, temp_client)
           print('beta', beta)
           # print('a', arrive_t)
           print('f', finish_t)
           # for i in range(args.num_client):
           #     if (len(finish_t[i])-1) % 5 == 0:
           #         print('f', finish_t)
           previous_c = temp_client
           previous_p = temp_point
           opt_seq_index, temp_client, temp_point, count, c = optimize_sequence(args, seqs, k, p, beta, j, arrive_t, finish_t, temp_client, temp_point, sum_mid_data)
           k = opt_seq_index
           seq = seqs[k]
           beta = compute_beta(args, seqs, k, c, beta, count)
           arrive_t = arrive_time(args, seq, j, finish_t, beta, arrive_t, temp_client, temp_point)
           finish_t = finish_time1(args, seq, k, beta, sum_mid_data, j, temp_client, temp_point, arrive_t, finish_t, agg)
       c_max_f = [0] * args.num_client
       for i in range(args.num_client):
           c_max_f[i] = finish_t[i][-1]
       min_finish_time = max(c_max_f)
       return min_finish_time


def main():
    args = args_parser()
    min_time = point_main(args)
    print('non_dynamic min_time', min_time)

if __name__ == '__main__':
    main()
