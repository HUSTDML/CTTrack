import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from lib.test.vot20.cttrack_vot20 import run_vot_exp

def main():
    run_vot_exp('cttrack_online', 'baseline_L', vis=False)