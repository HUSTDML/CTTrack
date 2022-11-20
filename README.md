# CT-Track
This is for the AAAI2023 blind review

## Install the environment
```
conda create -n cttrack python=3.7
conda activate cttrack
bash install.sh
```

## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${CT-TRACK_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Train CT-Track-B
```
python tracking/train.py --script cttrack --config baseline --save_dir . --mode single
python tracking/train.py --script cttrack_online --config baseline --save_dir . --mode single --script_prv cttrak --config_prv baseline  
```
## Train CT-Track-L
```
python tracking/train.py --script cttrack --config baseline_L --save_dir . --mode single
python tracking/train.py --script cttrack_online --config baseline_L --save_dir . --mode single --script_prv cttrak --config_prv baseline  
```
## Test CT-Track-B
- OTB2015
```
python tracking/test.py cttrack baseline --dataset otb --threads 32
```
- UAV123
```
python tracking/test.py cttrack baseline --dataset uav --threads 32
```
- LaSOT
```
python tracking/test.py cttrack baseline --dataset lasot --threads 32
```
- GOT10K-test
```
python tracking/test.py cttrack baseline --dataset got10k_test --threads 32
```
- TrackingNet
```
python tracking/test.py cttrack baseline --dataset trackingnet --threads 32
```
## Test CT-Track-L
- OTB2015
```
python tracking/test.py cttrack baseline_L --dataset otb --threads 32
```
- UAV123
```
python tracking/test.py cttrack baseline_L --dataset uav --threads 32
```
- LaSOT
```
python tracking/test.py cttrack baseline_L --dataset lasot --threads 32
```
- GOT10K-test
```
python tracking/test.py cttrack baseline_L --dataset got10k_test --threads 32
```
- TrackingNet
```
python tracking/test.py cttrack baseline_L --dataset trackingnet --threads 32
```
