# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Part-Listing I-JEPA: Main entry point
# Analogous to main.py but for part-listing training

import argparse
import sys
import yaml
from src.part_listing_train import main as part_listing_main


def parse_args():
    parser = argparse.ArgumentParser(
        description='Part-Listing I-JEPA Training',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--fname', type=str,
        help='name of config file to load',
        default='configs/part_listing_vitb16_ep100.yaml')
    parser.add_argument(
        '--devices', type=str, nargs='+', default=['cuda:0'],
        help='which devices to use on local machine')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        print('loaded params...')

    part_listing_main(args=params)
