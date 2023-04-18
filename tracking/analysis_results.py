import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.data import get_dataset
from lib.test.tracker import trackerlist

trackers = []

def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for analysis')
    # for analysis
    parser.add_argument('--script', type=str, default='cttrack',help='training script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--dataset_name', type=str, default='lasot',help='dataset of analysis')
    args = parser.parse_args()

    return args

args=parse_args()

trackers.extend(trackerlist(name=args.script, parameter_name=args.config, dataset_name=args.dataset_name,
                            run_ids=0, display_name=args.script))
dataset = get_dataset(args.dataset_name)
print_results(trackers, dataset, args.dataset_name, merge_results=False, plot_types=('success', 'prec', 'norm_prec'))
print_per_sequence_results(trackers, dataset, report_name="debug")

