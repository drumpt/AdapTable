for pretrain_epochs in 10 30 50 100
do
    for seed in 0 1 2
    do
        python main.py \
            meta_dataset=tableshift \
            dataset=heloc \
            pretrain_epochs=${pretrain_epochs} \
            mask_ratio=0.3 \
            seed=${seed} \
            test_lr=1e-3 \
            num_steps=20 \
            out_dir=heloc/pretrain_epochs_${pretrain_epochs}_seed_${seed}_hparams
    done
done