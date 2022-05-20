from icevision.all import *
from icevision.models.checkpoint import *

data_dir = Path('fire-dataset/train')
images_dir = data_dir / 'images'
annotations_dir = data_dir / 'annotations'

image_files = [i for i in images_dir.glob('*')]
annotation_files = [i for i in annotations_dir.glob('*')]

class_map = ClassMap(['fire'])
parser = parsers.VOCBBoxParser(annotations_dir=annotations_dir, images_dir=images_dir, class_map=class_map)
parser.class_map

data_splitter = RandomSplitter((.8, .2))
train_records, valid_records = parser.parse(data_splitter)

show_records(train_records[:3], ncols=3, class_map=class_map)

presize = 512
size = 384

train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=size, presize=presize), tfms.A.Normalize()])
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=size), tfms.A.Normalize()])

train_ds = Dataset(train_records, train_tfms)
valid_ds = Dataset(valid_records, valid_tfms)

samples = [train_ds[3] for _ in range(6)]
show_samples(samples, ncols=3, class_map=class_map)

model_type = models.ultralytics.yolov5
backbone = model_type.backbones.small

# The yolov5 model requires an img_size parameter
extra_args = {}
extra_args['img_size'] = size

# Instantiate the mdoel
model = model_type.model(backbone=backbone(pretrained=True), num_classes=len(parser.class_map), **extra_args)

# Data Loaders
train_dl = model_type.train_dl(train_ds, batch_size=8, num_workers=4, shuffle=True)
valid_dl = model_type.valid_dl(valid_ds, batch_size=8, num_workers=4, shuffle=False)

model_type.show_batch(first(valid_dl), ncols=4)

metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]

"""#Training"""

learn = model_type.fastai.learner(dls=[train_dl, valid_dl], model=model, metrics=metrics)

learn.lr_find()

learn.fine_tune(20, 1e-4, freeze_epochs=1)

learn.fine_tune(100, 3e-3, freeze_epochs=1)

model_type.show_results(model, valid_ds, detection_threshold=.5)

"""#Save"""

checkpoint_path = "./models/model_checkpoint.pth"

save_icevision_checkpoint(model,
                          model_name='ultralytics.yolov5', 
                          backbone_name='small',
                          img_size=640,
                          classes=parser.class_map.get_classes(),
                          filename=checkpoint_path,
                          meta={'icevision_version': '0.12.0'})