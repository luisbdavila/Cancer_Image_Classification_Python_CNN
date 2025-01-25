# DL Project

Project trying to classify cancer images (benign or malign, and cancer type) using CNN using keras/tensorflow.

To understand what is happening see the notebooks in this order:
1) EDA.
2) Preprocesing.
3) Benchmark.
4) Binary.
5) Multiclass.
6) Experiment.

Data: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

```plaintext
root/
├── DeepLearning24_25/                         # Data folder given with the images
├── Group_1/                                   # Our solution.
│   ├── Notebooks/                             # Folder to with jupyter notebooks.
│   │   ├── EDA.ipynb                          # Initial data exploration.
│   │   ├── Preprocessing.ipynb                # Initial preprocessing.
│   │   ├── Benchmark.ipynb                    # Our minimal accepted solution.
│   │   ├── Binary_classification.ipynb        # Binary classification model.
│   │   ├── Multiclass_classification.ipynb    # Multiclass classification model.
│   │   ├── Experiment.ipynb                   # Final model testing.
│   │   ├── utils/                             # Folder to store the functions.
│   │   │   ├── tools.py                       # Functions used.
│   ├── Data_Images/                           # Folder with the images and labels compressed to train our models.
│   │   ├── LabelEncoder_Bi_MC.pkl             # Label encoder for binary and multiclass arrays.
│   │   ├── Test_img_yEncode_Bi_MC_Mag.pkl     # Arrays used for the test set.
│   │   ├── Train_HP_img_yEncode_Bi_MC_Mag.pkl # Arrays used for the train set on hold-out method.
│   │   ├── Train_img_yEncode_Bi_MC_Mag.pkl    # Arrays used for the train set on cross-valiation.
│   │   ├── Val_HP_img_yEncode_Bi_MC_Mag.pkl   # Arrays used for the val set on hold-out method.
│   │   ├── Ylabels_Bi_MC.pkl                  # Labels for binary and multiclass.
│   ├── logs/                                  # Folder for the keras tuner.
├───├── Deep Learning Group Project24_25.pdf   # Project description.
├───├── Group_1_Report.pdf                     # Project report.
├───├── image_data_clean.csv                   # Our paths and labels with no misisng values.
├───├── readme.MD                              # Project structure information.
