# AdapTable

## Environmental Setup
```
conda create -n adaptable python=3.8.16 -y
conda activate adaptable
pip install -r requirements.txt
```

## Run
- edit conf/config.yaml properly.
- To run adaptable, run the following:
```
python main.py \
    benchmark=tableshift \
    dataset=heloc \
    shift_type=None \
    shift_severity=1 \
    --config-name ours_mlp.yaml
```
