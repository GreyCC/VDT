# Domain Transfer Generative Model for New Face Generation 

This is the program code of our accepted ICIP 2025 paper:

Chun-Chuen Hui, Wan-Chi Siu, and H. Anthony Chan, "Domain Transfer Generative Model for New Face Generation," in Proceedings of the International Conference on Image Processing, pp.1-6, September 2025. 


## Dataset
For training, you need to download the FFHQ dataset, you can follow the website: https://github.com/NVlabs/ffhq-dataset

## Training
Train the model by following the command lines below
```
python train.py --device cuda:0 --save_folder VDT_FFHQ --data_path YOUR_DATA_PATH --batch_size 8
```

## Recall
After the training you can run the following command to generate new face images
```
python eval.py --device cuda:0 --save_folder VDT_eval --load_path YOUR_MODEL_PATH 
```
