from mmengine.dataset import BaseDataset
from mmdet.registry import DATASETS
import numpy as np
import json

@DATASETS.register_module()
class DentexDataset(BaseDataset):
    CLASSES = ('1', '2', '3', '4')  # Update with your class names

    def __init__(self, ann_file, pipeline, img_prefix='', **kwargs):
        self.img_prefix = img_prefix
        self.ann_file = ann_file

        super().__init__(ann_file=ann_file, pipeline=pipeline, **kwargs)

    def load_data_list(self):
        with open(self.ann_file, 'r') as file:
            data = json.load(file)
        data_infos  = []
        for img_info in data['images']:
            img_id = img_info['id']
            filename = self.img_prefix + img_info['file_name']
            width = img_info['width']
            height = img_info['height']

            instances = []
            for ann in filter(lambda x: x['image_id'] == img_id, data['annotations']):
                bbox = self._convert_to_xyxy(np.array(ann['bbox'], dtype=np.float32))
                bbox_label = ann['category_id_1']
                mask = ann['segmentation']
                extra_anns = [ann['category_id_1'], ann['category_id_2'], ann['category_id_3']]

                instances.append({
                    "bbox": bbox,
                    "bbox_label": bbox_label,
                    "mask": mask,
                    "extra_anns": extra_anns
                })
                

            data_infos.append({
                "img_id": img_id,
                "img_path": filename,
                "height": height,
                "width": width,
                "instances": instances
            })
        return data_infos



    def _convert_to_xyxy(self, bbox):
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        return [x_min, y_min, x_max, y_max]
