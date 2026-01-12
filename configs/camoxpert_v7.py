_base_ = ["icod_train.py"]

# --- V7 CONFIG (B5 + Spectral MoE) ---
__BATCHSIZE = 8
__NUM_EPOCHS = 120 

train = dict(
    batch_size=__BATCHSIZE,
    num_workers=4,
    use_amp=True,
    num_epochs=__NUM_EPOCHS,
    epoch_based=True,
    lr=0.00005,
    optimizer=dict(
        mode="adamw",
        set_to_none=True,
        group_mode="finetune",
        cfg=dict(
            weight_decay=1e-4,
            diff_factor=0.1,
        ),
    ),
    sche_usebatch=True,
    scheduler=dict(
        warmup=dict(num_iters=1000),
        mode="cos",
        cfg=dict(
            lr_decay=0.9,
            min_coef=0.01,
        ),
    ),
    data=dict(
        shape=dict(h=448, w=448),
        names=["cod10k_tr"],
        root_path="C:/Users/GPUVM/thesis/ZoomNext/data/COD-TestDataset/COD-TestDataset", 
        train_image_path="Train/Image",
        train_mask_path="Train/GT_Object",
        test_image_path="Test/Image",
        test_mask_path="Test/GT_Object",
    ),
)

test = dict(
    batch_size=__BATCHSIZE,
    data=dict(
        shape=dict(h=448, w=448),
        names=["cod10k_te"],
    ),
)