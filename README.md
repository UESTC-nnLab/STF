# Spatio-temporal fusion for moving small target detection on satellite sequence images

## Build
You can follow [Centernet](https://github.com/xingyizhou/CenterNet) to build the code.

## Dataset
You can get the dataset in [DSFNet](https://github.com/ChaoXiao12/Moving-object-detection-DSFNet), and you should  download the dataset and put it to the data folder.

## Train
```
python train.py --model_name STFNet --gpus 0,1 --lr 1.25e-4 --lr_step 30,45 --num_epochs 70 --batch_size 2 --val_intervals 5  --test_large_size True --datasetname rsdata --data_dir  {data_path}
```
## Test
```
python test.py --model_name STFNet --gpus 0 --load_model {checkpoint_path} --test_large_size True --datasetname rsdata --data_dir {data_path}
```
## Visualization
```
python test.py --model_name STFNet --gpus 0 --load_model {checkpoint_path} --test_large_size True --show_results True --datasetname rsdata --data_dir {data_path} 
```
## Evaluation
```
1.python testSaveMat.py
2.python evaluation.py
```

## References
C. Xiao, Q. Yin, X. Ying, R. Li, S. Wu, M. Li, L. Liu, W. An, Z. Chen,
Dsfnet: Dynamic and static fusion network for moving object detection
in satellite videos, IEEE Geoscience and Remote Sensing Letters 19
(2022) 1–5.