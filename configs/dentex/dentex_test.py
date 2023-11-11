_base_ = [
    '../_base_/default_runtime.py'
]

# Dataset configuration
dataset_type = 'DentexDataset'
data_root = '/storage/dentex/train/training_data/quadrant-enumeration-disease/'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1080, 1080), keep_ratio=False),
    
]

data = dict(
    test=dict(
        type=dataset_type,
        ann_file='/storage/dentex/train/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json',
        img_prefix='/storage/dentex/train/training_data/quadrant-enumeration-disease/xrays/',
        pipeline=test_pipeline)
)

