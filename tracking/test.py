import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)


from lib.test.data import get_dataset
from lib.test.running import run_dataset
from lib.test.tracker import Tracker

def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                num_gpus=8,report_name=None):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """
    dataset = get_dataset(dataset_name)
    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id =run_id, report_name=report_name)]

    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)

def main():

    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--tracker_name', type=str, default='cttrack', help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, default='mixattn_L',help='Name of config file.')
    parser.add_argument('--runid', type=int, default=0, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='lasot', help='Name of dataset (otb, nfs, got10k_test,uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help ='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=10, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=1)
    args = parser.parse_args() 

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                args.threads, num_gpus=args.num_gpus)

if __name__ == '__main__':

    main()
