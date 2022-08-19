import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from lib.test.vot20.cttrack_vot20 import run_vot_exp

def main():
    run_vot_exp('cttrack_online', 'mixattn', vis=False)