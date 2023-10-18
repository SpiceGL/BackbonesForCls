# model settings

model_cfg = dict(
    backbone=dict(type='MobileViT', arch='small'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=640,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# train
data_cfg = dict(
    batch_size = 16,
    num_workers = 2,
    train = dict(
        pretrained_flag = True,
        pretrained_weights = 'pretrain/mobilevit-small_3rdparty_in1k_20221018-cb4f741c.pth',
        freeze_flag = False,
        freeze_layers = ('backbone',),
        epoches = 100,
    ),
    test=dict(
        ckpt = 'pretrain/mobilevit-small_3rdparty_in1k_20221018-cb4f741c.pth',
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'confusion'],
        metric_options = dict(
            topk = (1,5),
            thrs = None,
            average_mode='none'
    )
    )
)

# batch 32
# lr = 0.1 *32 /256
# optimizer
optimizer_cfg = dict(
    type='SGD',
    lr=0.1 * 32/256,
    momentum=0.9,
    weight_decay=1e-4)

# learning 
lr_config = dict(type='StepLrUpdater', step=2, gamma=0.973, by_epoch=True)
#lr_config = dict(type='StepLrUpdater', warmup='linear', warmup_iters=500, warmup_ratio=0.25,step=[30,60,90])

