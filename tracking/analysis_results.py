import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.data import get_dataset
from lib.test.tracker import trackerlist

trackers = []

trackers.extend(trackerlist(name='cttrack_online', parameter_name='mixattn', dataset_name='lasot',
                            run_ids=0, display_name='cttrack_online'))
dataset = get_dataset('lasot')
print_results(trackers, dataset, 'lasot', merge_results=False, plot_types=('success', 'prec', 'norm_prec'))
print_per_sequence_results(trackers, dataset, report_name="debug")

