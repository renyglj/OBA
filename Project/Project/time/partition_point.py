import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter
import heapq

# fclient1 = [3.733241714, 10.37337276, 10.94490835, 14.07115686, 18.16329516, 18.45793872, 22.42133741, 28.32225477, 34.14074412, 34.40372916, 42.01877978, 56.92792802, 71.81076946, 72.11538053, 85.74500675, 99.28604637, 112.7419802, 112.8601723, 113.5121583, 114.0642839, 114.3626829]
# fclient2 = [0.5*i for i in fclient1]
# fclient3 = [0.7*i for i in fclient1]
# fclient4 = [0.9*i for i in fclient1]
# fclient5 = [1.1*i for i in fclient1]
# fclient6 = [1.3*i for i in fclient1]
# fclient7 = [1.5*i for i in fclient1]
# fclient8 = [1.7*i for i in fclient1]
# fclient9 = [0.05*i for i in fclient1]
# fclient10 = [0.1*i for i in fclient1]
# cfw_time = [[fclient1], [fclient2], [fclient3], [fclient4], [fclient5], [fclient6], [fclient7], [fclient8], [fclient9], [fclient10]]
#
# sfw_time = [15.74715936, 14.84341609, 13.75992175, 13.5046979, 12.7117063, 11.82491775, 11.60618928, 10.87255931, 10.01874134, 9.175249818, 8.929121744, 8.011226927, 6.653599851, 5.476118388, 5.216449274, 4.057106917, 2.801544364, 1.584089134, 1.354580489, 0.927146507, 0.45035992]
#
# mid_data = [12.845112, 12.845112, 3.21132, 6.422584, 6.422584, 1.605688, 3.21132, 3.21132, 3.21132, 0.802872, 1.605688, 1.605688, 1.605688, 0.401464, 0.401464, 0.401464, 0.401464, 0.100408, 0.01644, 0.01644, 0.01644]
from option import args_parser


class Clients():
        def __init__(self, id, cfw_time, cbw_time, c_time, mid_data, para):
                self.id = id
                self.cfw_time = cfw_time
                self.cbw_time = cbw_time
                self.c_time = c_time
                self.mid_data = mid_data
                self.para = para
                self.iter = 0

class Server():
        def __init__(self, sfw_time, mid_data):
                self.sfw_time = sfw_time
                self.mid_data = mid_data

class Tlist():
        def __init__(self, id, point):
                self.id = id
                self.point = point
        def initialize(self, id, point):
                list = Tlist(id, point)
                return list

def initialize(args):
        server = Server(sfw_time=args.sfw_time, mid_data=args.mid_data)
        clients = []
        for i in range(args.num_client):
                clients.append(Clients(id=i,
                                       cfw_time=args.cfw_time[i],
                                       cbw_time=args.cbw_time[i],
                                       c_time=args.c_time[i],
                                       mid_data=args.mid_data[i],
                                       para=args.para[i]))
        return clients, server

def partition_point(args):
        clients, server = initialize(args)
        tlist = []
        B = args.bandwidth
        for i, client in enumerate(clients):
                t = []
                # print('device {}'.format(i+1))
                for r in range(0, 21):
                        t.append(clients[i].c_time[0][r]+server.sfw_time[r+1]+((2*server.mid_data[r])/B)*1000+(clients[i].para[0][r]/B)*2*1000)
                        # timetxt = str(t)
                        # with open('point.txt', 'a') as file_handle:
                        #         file_handle.write(timetxt)
                        #         file_handle.write('\n')
                print(t)
                min_number = heapq.nsmallest(4, t)  #nsmallest()找出列表中最小的n个元素
                min_index = []
                for j in min_number:
                        index = t.index(j)
                        min_index.append(index)
                        t[index] = 0
                # print(min_index)
                list = Tlist(id=i, point=min_index)
                # list1 = list.initialize(i, min_index)
                # print(list.id, list.time)
                tlist.append(list.point)

        return tlist


def main():
        args = args_parser()
        p = partition_point(args)
        for i, point in enumerate(p):
                print(p[i])
        # print(p[0])

if __name__ == '__main__':
        main()
