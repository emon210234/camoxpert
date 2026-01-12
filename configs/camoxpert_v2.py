_base_ = ["icod_train.py"]

# --- CamoXpert V2 CONFIGURATION ---
__BATCHSIZE = 20  # Safe for 24GB VRAM
__NUM_EPOCHS = 120 # Full training cycle

train = dict(
    batch_size=__BATCHSIZE,
    num_workers=4,
    use_amp=True,
    num_epochs=__NUM_EPOCHS,
    epoch_based=True,
    lr=0.0001,
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
    # Path Configuration (Windows)
    data=dict(
        shape=dict(h=384, w=384),
        names=["cod10k_tr"],
        # Ensure this matches your actual dataset path
        root_path="D:/emon210234/ZoomNext/data/archive/COD10K-v3", 
        train_image_path="Train/Image",
        train_mask_path="Train/GT_Object",
        test_image_path="Test/Image",
        test_mask_path="Test/GT_Object",
    ),
)

test = dict(
    batch_size=__BATCHSIZE,
    data=dict(
        shape=dict(h=384, w=384),
        names=["cod10k_te"],
    ),
)