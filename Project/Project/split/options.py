import argparse

import torch.cuda


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        help='name of the dataset'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='vgg-16',
        help='name of the model'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='learning rate of the SGD when trained on the client'
    )
    parser.add_argument(
        '--lr_decay',
        type=float,
        default=0,
        help='lr decay rate'
    )
    parser.add_argument(
        '--lr_decay_epoch',
        type=int,
        default=1,
        help='lr decay epoch'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='SGD momentum'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.001,
        help='the weight decay rate'
    )
    parser.add_argument(
        '--num_clients',
        type=int,
        default=10,
        help='number of the clients'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='data',
        help='dataset root folder'
    )
    parser.add_argument(
        '--iid',
        type=int,
        default=0,
        help='distribution of data'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='select gpu'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='random seed'
    )
    parser.add_argument(
        '--num_communication',
        type=int,
        default=100,
        help='number of iteration on edge'
    )
    parser.add_argument(
        '--num_local_update',
        type=int,
        default=5,
        help='number of iteration on client'
    )
    parser.add_argument(
        '--show_dis',
        type=int,
        default=1
    )

    args = parser.parse_args()
    # args.cuda = torch.cuda.is_available()
    args.cuda = False
    return args
