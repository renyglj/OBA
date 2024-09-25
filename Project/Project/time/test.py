from option import args_parser
from non_partition import point_main as non_partition_point_main
from fixed_point import point_main as fixed_point_point_main
from random_point import point_main as random_point_point_main
from average import  point_main as average_point_main
from random_bandwidth import point_main as random_bandwidth_point_main
from OBA import point_main as OBA_point_main

def main():
    args = args_parser()

    no_partitioned_min_time = non_partition_point_main(args)
    print('no_partitioned min_time', no_partitioned_min_time)

    fixed_point_min_time = fixed_point_point_main(args)
    print('fixed_point min_time', fixed_point_min_time)

    random_point_min_time = random_point_point_main(args)
    print('random_point min_time', random_point_min_time)

    average_min_time = average_point_main(args)
    print('average min_time', average_min_time)

    random_bandwidth_min_time = random_bandwidth_point_main(args)
    print('random_bandwidth min_time', random_bandwidth_min_time)

    OBA_min_time = OBA_point_main(args)
    print('OBA min_time', OBA_min_time)

    print('汇总：')
    print('no_partitioned min_time', no_partitioned_min_time)
    print('fixed_point min_time', fixed_point_min_time)
    print('random_point min_time', random_point_min_time)
    print('average min_time', average_min_time)
    print('random_bandwidth min_time', random_bandwidth_min_time)
    print('OBA min_time', OBA_min_time)

if __name__ == '__main__':
    main()