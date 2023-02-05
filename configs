_base_ = [
    '../_base_/models/resnet18.py', '../_base_/datasets/imagenet_bs32.py',
     '../_base_/schedules/imagenet_bs256.py' ,'../_base_/default_runtime.py'
]

model = dict(head=dict(num_classes=5, topk = (1, )))
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        
        data_prefix='data/flower_dataset/train',
        ann_file = 'data/flower_dataset/train.txt',
        classes = 'data/flower_dataset/classes.txt'
        ),
    val=dict(
        
        data_prefix='data/flower_dataset/val',
        ann_file = 'data/flower_dataset/val.txt',
        classes = 'data/flower_dataset/classes.txt'
        ),
        )
    

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step',step=[100,120])
runner = dict(type='EpochBasedRunner', max_epochs=150)

evaluation = dict(metric_options={'topk': (1, )})
load_from ='/home/lei/anaconda3/envs/mmlab1/mmclassification/ckeckpoints/resnet18_batch256_imagenet_20200708-34ab8f90.pth'


      
