import argparse
def args_parser():
    parser = argparse.ArgumentParser()
    #客户端正向传播各层作为分割点的计算时间
    fclient1 = [0, 17.0807038,	49.46321597,	53.20221877,	67.69265554,	89.35745718,	91.17233056,	101.8983115,	119.4603848,	136.5854308,	137.5265873,	147.732757,	163.3540806,	178.7936298,	179.3405156,	186.6049343,	193.6189807,	200.4399819,	200.5928201,	200.9502306,	201.2369588,	201.3715761]
    fclient2 = [0.5 * i for i in fclient1]
    fclient3 = [0.7 * i for i in fclient1]
    fclient4 = [0.9 * i for i in fclient1]
    fclient5 = [1.1 * i for i in fclient1]
    fclient6 = [1.3 * i for i in fclient1]
    fclient7 = [0.3 * i for i in fclient1]
    fclient8 = [1.7 * i for i in fclient1]
    fclient9 = [1.5 * i for i in fclient1]
    fclient10 = [0.1 * i for i in fclient1]
    fclient11 = [0, 1.446494921, 3.631371443, 3.947689664, 5.181452036, 6.696676652, 6.881962292, 8.31279072, 10.16874788, 11.8525039, 11.96433915, 13.98889082, 17.82695465, 21.7823189,	21.86297762, 25.40023554, 28.77443951, 32.23903496, 32.29096334, 32.41662719, 32.54917063, 32.60823198]
    fclient12 = [0.5 * i for i in fclient11]
    fclient13 = [0.7 * i for i in fclient11]
    fclient14 = [0.9 * i for i in fclient11]
    fclient15 = [1.1 * i for i in fclient11]
    fclient16 = [1.3 * i for i in fclient11]
    fclient17 = [0.3 * i for i in fclient11]
    fclient18 = [1.7 * i for i in fclient11]
    fclient19 = [1.5 * i for i in fclient11]
    fclient20 = [0.1 * i for i in fclient11]
    # cfw_time = [[fclient1], [fclient2], [fclient3], [fclient4], [fclient5]]
    # cfw_time = [[fclient1], [fclient2], [fclient3], [fclient4], [fclient5], [fclient6], [fclient7], [fclient8],[fclient9], [fclient10]]
    cfw_time = [[fclient1], [fclient2], [fclient3], [fclient4], [fclient5], [fclient6], [fclient7], [fclient8],
                [fclient9], [fclient10], [fclient11], [fclient12], [fclient13], [fclient14], [fclient15], [fclient16],
                [fclient17], [fclient18],
                [fclient19], [fclient20]]
    # 客户端反向传播各层作为分割点的计算时间
    bclient1 = [0, 31.93382189,	97.51007236,	102.1731716,	129.7973729,	172.6704765,	174.9232742,	196.2993726,	231.3243884,	269.7921765,	270.9561976,	292.3532112,	328.1842178,	363.8119812,	364.4342745,	385.0352465,	407.2159381,	430.9318518,	431.0316703,	431.9777046,	432.4418808,	432.593398]
    bclient2 = [0.5 * i for i in bclient1]
    bclient3 = [0.7 * i for i in bclient1]
    bclient4 = [0.9 * i for i in bclient1]
    bclient5 = [1.1 * i for i in bclient1]
    bclient6 = [1.3 * i for i in bclient1]
    bclient7 = [0.3 * i for i in bclient1]
    bclient8 = [1.7 * i for i in bclient1]
    bclient9 = [1.5 * i for i in bclient1]
    bclient10 = [0.1 * i for i in bclient1]
    bclient11 = [0, 2.286746792, 6.74200132, 6.997218683, 8.889704825, 11.4666185, 11.57597642, 14.10854669, 18.15350689, 22.28824022, 22.43939001, 28.02988896, 39.10097337, 50.02845056, 50.25240291, 60.34477121, 70.51160685, 80.50294528, 80.56920897, 81.09553107, 81.51511329, 81.75445091]
    bclient12 = [0.5 * i for i in bclient11]
    bclient13 = [0.7 * i for i in bclient11]
    bclient14 = [0.9 * i for i in bclient11]
    bclient15 = [1.1 * i for i in bclient11]
    bclient16 = [1.3 * i for i in bclient11]
    bclient17 = [0.3 * i for i in bclient11]
    bclient18 = [1.7 * i for i in bclient11]
    bclient19 = [1.5 * i for i in bclient11]
    bclient20 = [0.1 * i for i in bclient11]
    # cbw_time = [[bclient1], [bclient2], [bclient3], [bclient4], [bclient5]]
    # cbw_time = [[bclient1], [bclient2], [bclient3], [bclient4], [bclient5], [bclient6], [bclient7], [bclient8], [bclient9], [bclient10]]
    cbw_time = [[bclient1], [bclient2], [bclient3], [bclient4], [bclient5], [bclient6], [bclient7], [bclient8],
                [bclient9], [bclient10], [bclient11], [bclient12], [bclient13], [bclient14], [bclient15], [bclient16],
                [bclient17], [bclient18],
                [bclient19], [bclient20]]
    # 客户端上每层作为分割点正向+反向 时间
    client1 = [0, 49.01452569,	146.9732883,	155.3753904,	197.4900284,	262.0279336,	266.0956048,	298.1976841,	350.7847733,	406.3776073,	408.4827849,	440.0859682,	491.5382984,	542.605611,	543.7747901,	571.6401808,	600.8349188,	631.3718337,	631.6244904,	632.9279353,	633.6788396,	633.9649741]
    client2 = [0.5 * i for i in client1]
    client3 = [0.7 * i for i in client1]
    client4 = [0.9 * i for i in client1]
    client5 = [1.1 * i for i in client1]
    client6 = [1.3 * i for i in client1]
    client7 = [0.3 * i for i in client1]
    client8 = [1.7 * i for i in client1]
    client9 = [1.5 * i for i in client1]
    client10 = [0.1 * i for i in client1]
    client11 = [0, 3.733241714, 10.37337276, 10.94490835, 14.07115686, 18.16329516, 18.45793872, 22.42133741, 28.32225477, 34.14074412, 34.40372916, 42.01877978, 56.92792802, 71.81076946, 72.11538053, 85.74500675, 99.28604637, 112.7419802, 112.8601723, 113.5121583, 114.0642839, 114.3626829]
    client12 = [0.5 * i for i in client11]
    client13 = [0.7 * i for i in client11]
    client14 = [0.9 * i for i in client11]
    client15 = [1.1 * i for i in client11]
    client16 = [1.3 * i for i in client11]
    client17 = [0.3 * i for i in client11]
    client18 = [1.7 * i for i in client11]
    client19 = [1.5 * i for i in client11]
    client20 = [0.1 * i for i in client11]
    # c_time = [[client1], [client2], [client3], [client4], [client5]]
    # c_time = [[client1], [client2], [client3], [client4], [client5], [client6], [client7], [client8], [client9], [client10]]
    c_time = [[client1], [client2], [client3], [client4], [client5], [client6], [client7], [client8], [client9],
              [client10],
              [client11], [client12], [client13], [client14], [client15], [client16], [client17], [client18],
              [client19], [client20]]
    # 服务器上每层作为分割点 正向+反向 时间
    sfw_time = [15.74715936, 14.84341609, 13.75992175, 13.5046979, 12.7117063, 11.82491775, 11.60618928, 10.87255931,
                10.01874134, 9.175249818, 8.929121744, 8.011226927, 6.653599851, 5.476118388, 5.216449274, 4.057106917,
                2.801544364, 1.584089134, 1.354580489, 0.927146507, 0.45035992, 0]

    # mid_data = [12.845112, 12.845112, 3.21132, 6.422584, 6.422584, 1.605688, 3.21132, 3.21132, 3.21132, 0.802872,
    #             1.605688, 1.605688, 1.605688, 0.401464, 0.401464, 0.401464, 0.401464, 0.100408, 0.01644, 0.01644,
    #             0.01644, 0]
    # 中间数据量
    mid_data = [16.00005341, 16.00005341, 4.000053406, 8.000053406,	8.000053406, 2.000053406, 4.000053406, 4.000053406,	4.000053406, 1.000053406, 2.000053406, 2.000053406,	2.000053406, 0.500053406, 0.500053406, 0.500053406, 0.500053406, 0.125053406, 0.125053406, 0.062553406, 0.002494812, 0]
    #模型参数量
    cpara = [0, 0.001831055, 0.03717041, 0.03717041, 0.107849121, 0.248840332, 0.248840332, 0.530822754, 1.094055176, 1.657287598, 1.657287598, 2.783752441, 5.035217285, 7.286682129, 7.286682129, 9.538146973, 11.78961182, 14.04107666, 14.04107666, 14.29156494, 14.41680908, 14.41926003]
    pclient1 = cpara
    pclient2 = cpara
    pclient3 = cpara
    pclient4 = cpara
    pclient5 = cpara
    pclient6 = cpara
    pclient7 = cpara
    pclient8 = cpara
    pclient9 = cpara
    pclient10 = cpara
    pclient11 = cpara
    pclient12 = cpara
    pclient13 = cpara
    pclient14 = cpara
    pclient15 = cpara
    pclient16 = cpara
    pclient17 = cpara
    pclient18 = cpara
    pclient19 = cpara
    pclient20 = cpara
    # para = [[pclient1], [pclient2], [pclient3], [pclient4], [pclient5]]
    # para = [[pclient1], [pclient2], [pclient3], [pclient4], [pclient5], [pclient6] , [pclient7], [pclient8], [pclient9], [pclient10]]
    para = [[pclient1], [pclient2], [pclient3], [pclient4], [pclient5], [pclient6], [pclient7], [pclient8], [pclient9],
            [pclient10],
            [pclient11], [pclient12], [pclient13], [pclient14], [pclient15], [pclient16], [pclient17], [pclient18],
            [pclient19], [pclient20]]
    parser.add_argument(
        '--cfw_time',
        type=list,
        default=cfw_time,
        help='execute time on client'
    )
    parser.add_argument(
        '--cbw_time',
        type=list,
        default=cbw_time,
        help='execute time on client'
    )
    parser.add_argument(
        '--c_time',
        type=list,
        default=c_time,
        help='execute time on client'
    )
    parser.add_argument(
        '--sfw_time',
        type=list,
        default=sfw_time,
        help='execute time on server'
    )
    parser.add_argument(
        '--mid_data',
        type=list,
        default=mid_data,
        help='middle data'
    )
    parser.add_argument(
        '--para',
        type=list,
        default=para,
        help='parametrs of layers'
    )
    parser.add_argument(
        '--num_comm',
        type=int,
        default=170,
        help='iterations on server'
    )
    parser.add_argument(
        '--num_client',
        type=int,
        default=12,
        help='number of clients'
    )
    parser.add_argument(
        '--bandwidth',
        type=float,
        default=10,
        help='total bandwidth'
    )

    args = parser.parse_args()
    return args