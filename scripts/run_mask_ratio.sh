for mask_ratio in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    for seed in 1 2
    do
        python main.py \
            meta_dataset=tableshift \
            dataset=heloc \
            mask_ratio=${mask_ratio} \
            seed=${seed} \
            test_lr=1e-3 \
            num_steps=20 \
            out_dir=heloc/mask_ratio_${mask_ratio}_seed_${seed}_hparams
    done
done