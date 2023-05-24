for seed in 0 1 2
    do
        python main.py \
            meta_dataset=tableshift \
            dataset=heloc \
            mask_ratio=0.3 \
            seed=${seed} \
            test_lr=1e-3 \
            num_steps=50 \
            out_dir=heloc/iter_step_seed_${seed}_hparams
    done
done