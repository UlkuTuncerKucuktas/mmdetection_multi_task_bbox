import mmcv
import matplotlib.pyplot as plt
from mmdet.registry import DATASETS
from mmengine.config import Config
import numpy as np

# Load the configuration and build the dataset
cfg = Config.fromfile('./configs/dentex/dentex_test.py')
dataset = DATASETS.build(cfg.data.test)

    
# Visualize first 10 data samples with extra annotations
for i, data in enumerate(dataset):
    
  
    if i >= 10:
        break

    img = data['img']
    gt =data['gt_bboxes']
    labels = np.array([inst['bbox_label'] for inst in data['instances']])
    extra_anns = [inst['extra_anns'] for inst in data['instances']]

    mmcv.imshow_bboxes(img, gt, thickness=2, show=False)
 
    
    # Plot extra annotations
    for bbox, anns in zip(gt, extra_anns):
        x_min, y_min, x_max, y_max = bbox
        label_text = f"Cat1: {anns[0]}, Cat2: {anns[1]}, Cat3: {anns[2]}"
        plt.text(x_min, y_min, label_text, color='green', fontsize=10)

    plt.imshow(mmcv.bgr2rgb(img))
    plt.axis('off')
    plt.savefig(f'./visualized_data_{i}.png')
    plt.close()
