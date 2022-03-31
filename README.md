
# Deep Learning, MLOps and Web App for Car Detection in Traffic Images 

This repository contains code to detect cars in traffic images with deep learning and its MLOps and web app. 

## Overview
- Detect and recognize cars in traffic workflow street view images with Deep Learning object detection (with PyTorch, Faster RCNN model)
- MLOps (Machine Learning Operations): Build a pipeline and end-to-end workflow from data preprocess to deployment and track the ML model (with Airflow and MLflow)
- Web app for car detection in traffic images on the cloud (with Flask and Docker)


## Pipeline
1. **Data preprocess:** Extract and transform original data.
2. **Hyper-parameter tuning:** Find optimal hyper-parameters with Ray Tune.
3. **Training:** Train and evaluate the model with train and validation dataset. Metric is mAP (mean average precision).
4. **Testing:** Evaluate the trained model with test dataset.
5. **Deployment:** Save the final model and app files for deployment. 


## Data
- Data for STREETS: A Novel Camera Network Dataset for Traffic Flow, Illinois Data Bank


## Reference
- PyTorch Documentation
- Apache Airflow Documentation
- MLflow Documentation
- Flask Documentation
- Ray Tune Documentation

