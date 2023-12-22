# Steel-Surface-Defect-Detection

Project on steel defect detection for the Artificial Intelligence in Manufacturing course offered in IIT Madras. This project involves performing semantic segmentation on pictures of steel surfaces to classify and localize the surface defects using the UNET architecture. The dataset for this project is taken from [Kaggle](https://www.kaggle.com/c/severstal-steel-defect-detection) which contains about 12000 train images and 6000 test images(more information is available on the Kaggle website).

## Files Present

```
rootdir
    ├── UNET_128x800_image/
    │    ├── dataloader.py
    │    ├── model.py
    │    └── train.py
    │
    ├── UNET_64x400_image/
    │    ├── dataloader.py
    │    ├── model.py
    │    └── train.py
    │
    ├── Model2_Modified_UNET/
    │    ├── dataset.py
    │    ├── model.py
    |    ├── lovasz_softmax.py
    |    ├── train.py
    │    └── README.md (experimental details)
    │
    ├── README.md
    └── Result_Tabulation.png
```

## About the Code:
The images in the dataset are of size 256x1600. The folder **UNET_128X800_image** contains the code which resizes the image to 128x800 and performs the segmentation task while the folder **UNET_64X400_image** contains the code which resizes the image to 64x400 and does the same as before. There are three codes dataloader.py, model.py and train.py in which train.py should be run to start the training.

Wandb is used here to log the losses, accurcies, scores and some input images along with its outputs. The wandb report can be referred [here](https://wandb.ai/manoj-s/Steel_Defect_Detection?workspace=user-manoj-s).

## Results Tabulation
![Results Tabulation](https://github.com/Manoj-152/Steel-Surface-Defect-Detection/blob/main/Result_Tabulation.png)

## Experiments

We also perform one other experiment focused on the speed and memory efficiency of the model. We were able to make the model twice as deeper, while having less than half the number of parameters as the above. Please note that this model does not perform as well the previous Unet but is over 20 times faster. All code relevant to this experiment can be found [here](./Model2_Modified_UNET/).

## Conclusion

- Built a suitable deep learning pipeline to localise and classify defects in steel. 
- Utilise a standard segmentation model UNet and experiment with its performance for various image sizes. 
- Experimented with another model focused on speed and memory efficiency. 
- The dataset is small and some of the annotations are incorrect, hence limiting the performance of our model. Utilising a larger and better dataset should help it perform better.
