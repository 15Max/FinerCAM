# XAI



### Repository
```
/XAI
│
├── references                       # Papers
│
├── data/
│   ├── train_data.mat               # Train dataset
│   └── test_data.mat                # Test dataset
├── models/
│   └── model.py                     # Defines model loading & fine-tuning
│
├── cam/
│   ├── gradcam.py                   # Classic Grad-CAM implementation
│   └── finercam.py                  # Finer-CAM modification
│
├── train.py                         # Fine-tunes a pretrained model
├── evaluate.py                      # Evaluates model accuracy
├── visualize_cam.py                 # Applies and visualizes CAMs
│
├── utils/
│   └── data.py                      # Dataloaders & preprocessing
│
├── req.yaml                         # Dependencies for the conda env
└── README.md                        # Project description and instructions
```