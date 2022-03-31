### common
PATH_ROOT = './dags/data'


### etl.py
PATH_ANNOTATIONS = "vehicleannotations/annotations"
PATH_IMAGE = 'images'
PATH_TARGETS = 'targets'

FILENAME_ANNOTATIONS = "vehicle-annotations.json"
FILENAME_TARGETS = 'targets.json'


### dl.py
PATH_WORKING_DIR = './dags'
PATH_MODELS = './dags/saved_model'
MODEL_TRAIN_NAME = 'trainded_model'
MODEL_SAVED_NAME = 'saved_model'

NUM_CLASSES = 2  # number of classes (0:background, 1:car)
MODEL_TYPE = 1  # Faster R-CNN model (0:Vannila, 1:Pre-trained)

MAX_EPOCHS = 10
IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.7

IS_ALL_DATA = True  # Use all data
DATA_SIZE = 1000
RATIO = (0.4, 0.4, 0.2)  # train/validation/test ratio

IS_SAVE = True  # save model

HYPER_PARAMETER = {
    'batch_size': [2, 8, 16],
    'learning_rate': [0.001, 0.0001, 0.00005],
    'epochs': [5, 10]
}


### deploy.py
PATH_DEPLOYMENT = './dags/deployment'
PATH_DEPLOYMENT_MODEL = 'model'
IMAGE_NAME = 'project'


### utils.py
NUM_AP_POINT = 11
