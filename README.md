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
   ${ÅRETRACK_ROOT}
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

## Train CT-Track
```
python tracking/train.py --script cttrack --config mixattn --save_dir . --mode single
python tracking/train.py --script cttrack_online --config mixattn --save_dir . --mode single --script_prv cttrak --config_prv mixattn  
```
## Test CT-Track
- OTB2015
```
python tracking/test.py cttrack mixattn --dataset otb --threads 32
```
- UAV123
```
python tracking/test.py cttrack mixattn --dataset uav --threads 32
```
- LaSOT
```
python tracking/test.py cttrack mixattn --dataset lasot --threads 32
```
- GOT10K-test
```
python tracking/test.py cttrack mixattn --dataset got10k_test --threads 32
```
- TrackingNet
```
python tracking/test.py cttrack mixattn --dataset trackingnet --threads 32
```
