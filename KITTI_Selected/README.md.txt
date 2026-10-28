# KITTI Selected Subset

This project uses a 20-frame subset of the KITTI Vision Benchmark Suite. The dataset is publicly available but not included in this repository.

## Download Instructions

1. Register for a free account at [cvlibs.net/datasets/kitti](http://www.cvlibs.net/datasets/kitti/)
2. Download the KITTI Object Detection benchmark files:
   - Left colour images
   - Camera calibration matrices
   - Training labels
3. Select 20 contiguous frames and place them in this folder:
KITTI_Selected/
├── images/ # 20 .jpg or .png files
├── labels/ # 20 .txt files (one per image)
└── calib/ # 20 .txt files (one per image)

4. Ensure the file naming matches across all three folders (e.g., `000001.jpg`, `000001.txt`).

## Label Format

Each line in a label `.txt` file:
class xmin ymin xmax ymax gt_distance


- `class` — object class name (e.g., `Car`)
- `xmin, ymin, xmax, ymax` — bounding box corners in pixel coordinates
- `gt_distance` — ground-truth distance to the object in metres

## Calibration Files

Each calibration `.txt` file contains the 3×3 camera intrinsic matrix **K** extracted from KITTI's standard calibration format.

## License

The KITTI dataset is distributed under **Creative Commons Attribution-NonCommercial-ShareAlike 3.0 (CC BY-NC-SA 3.0)**. Please respect the original authors' licensing terms when using the data.