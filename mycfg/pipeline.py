# dataloader pipeline
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)

net_size = (224, 224)   ### h, w

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomCropFixedSize', random_prop=1, cut_size_radio=10.0/110.0),
    # dict(type='To3cGray', random_prop=1),
    dict(type='Resize', size=net_size, backend='pillow'),  ## size=(h w)
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    # dict(type='MyLighting'),
    # dict(type='LetterBox', new_shape = net_size, color=(114, 114, 114), auto=False, scaleFill=False, stride=32),
    #dict(type='RandomResizedCrop', size=224, backend='pillow'),
    
    # dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
    # dict(type='ColorJitter', brightness=0.7, contrast=0.7, saturation=0.7),
    # dict(type='Lighting', **img_lighting_cfg),
    # dict(type='RandomGrayscale'),
    # dict(type='AutoAugment', policies=policies),
    # dict(type='Lighting', eigval= , eigvec),
    # dict(
    #     type='RandomErasing',
    #     erase_prob=0.25,
    #     mode='rand',
    #     min_area_ratio=0.02,
    #     max_area_ratio=1 / 10,
    #     fill_color=img_norm_cfg['mean'][::-1],
    #     fill_std=img_norm_cfg['std'][::-1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

no_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomCropFixedSize', random_prop=0, cut_size_radio=10.0/110.0),
    # dict(type='To3cGray', random_prop=1),
    dict(type='Resize', size=net_size, backend='pillow'),  ## size=(h w)
    # dict(type='LetterBox', new_shape=net_size, color=(114, 114, 114), auto=False, scaleFill=False, stride=32),
    #dict(type='RandomResizedCrop', size=224, backend='pillow'),
    # dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    # dict(type='RandomFlip', flip_prob=0.5, direction='vertical'),
    # dict(type='ColorJitter', brightness=0.5, contrast=0.5, saturation=0.5),
    # dict(type='RandomGrayscale'),
    # dict(type='AutoAugment', policies=policies),
    # dict(type='Lighting', eigval= , eigvec),
    # dict(
    #     type='RandomErasing',
    #     erase_prob=0.25,
    #     mode='rand',
    #     min_area_ratio=0.02,
    #     max_area_ratio=1 / 10,
    #     fill_color=img_norm_cfg['mean'][::-1],
    #     fill_std=img_norm_cfg['std'][::-1]),
    #dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    # dict(type='Lighting', **img_lighting_cfg),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomCropFixedSize', random_prop=0, cut_size_radio=10.0/110.0),
    # dict(type='To3cGray', random_prop=1),
    dict(type='Resize', size=net_size, backend='pillow'),  #size = (h,w)
    # dict(type='LetterBox', new_shape = net_size, color=(114, 114, 114), auto=False, scaleFill=False, stride=32),
    # dict(type='CenterCrop', crop_size=224)
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]