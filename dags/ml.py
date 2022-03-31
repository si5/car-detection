import copy
import os
import random
import time
from functools import partial

import mlflow
import ray
import torch
import torch.optim as optim
import torchvision
from ray import tune
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

import models
import settings
import utils
from dataset import Dataset


### Learning class
class Learner:
    def __init__(self):
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device: {}'.format(self.device))

    ### Transform
    def transform(self, path_data):

        # Data loading (images)
        transforms_image = transforms.Compose([torchvision.transforms.ToTensor()])
        dataset = Dataset(path_data, transform=transforms_image)

        # Data size
        if settings.IS_ALL_DATA:
            data_size = len(dataset)
        else:
            data_size = settings.DATA_SIZE

        train_size = int(data_size * settings.RATIO[0])
        valid_size = int(data_size * settings.RATIO[1])
        indices = [*range(len(dataset))]

        random.seed(0)
        random.shuffle(indices)

        # Dataset
        dataset_train = Subset(dataset, indices[:train_size])
        dataset_valid = Subset(dataset, indices[train_size : train_size + valid_size])
        dataset_test = Subset(dataset, indices[train_size + valid_size : data_size])

        return (dataset_train, dataset_valid, dataset_test)

    ### Train for one epoch
    def train_epoch(self, model, data_loader, optimizer, scheduler=None):
        model.train()
        loss_total = 0.0
        num_batches = 0

        for index, data_batch in enumerate(tqdm(data_loader)):
            images_batch, targets_batch = data_batch

            images_batch = [image.to(self.device) for image in images_batch]
            targets_batch = [
                {key: value.to(self.device) for key, value in target.items()}
                for target in targets_batch
            ]

            outputs = model(images_batch, targets_batch)
            loss = sum(outputs.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss
            num_batches += 1

        # average loss
        loss_mean = loss_total / num_batches

        return loss_total, loss_mean

    ### Evaluation
    def evaluation(self, model, data_loader):
        model.eval()
        num_batches = 0
        map_list = []

        with torch.no_grad():
            for index, data_batch in enumerate(tqdm(data_loader)):
                images_batch, targets_batch = data_batch

                images_batch = [image.to(self.device) for image in images_batch]
                targets_batch = [
                    {key: value.to(self.device) for key, value in target.items()}
                    for target in targets_batch
                ]

                outputs = model(images_batch, targets_batch)

                num_batches += 1

                for output, target in zip(outputs, targets_batch):
                    map = utils.mean_average_precision(
                        output, target, settings.IOU_THRESHOLD
                    )
                    map_list.append(map)

            # mean average precision
            map_mean = sum(map_list) / len(map_list)
            return map_mean

    ### Training
    def train(self, config, path_data):
        # Start time
        t0 = time.time()

        # Hyper parameter
        batch_size = config['batch_size']
        learning_rate = config['learning_rate']
        epochs = config['epochs']

        # Transform
        dataset_train, dataset_valid, _ = self.transform(path_data)

        # data loader
        data_loader_train = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=utils.collate_fn,
        )
        data_loader_valid = DataLoader(
            dataset_valid,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=utils.collate_fn,
        )

        # Model
        model_class = models.Model(settings.NUM_CLASSES)
        if settings.MODEL_TYPE == 0:
            model = model_class.frcnn_vanilla_model()
        elif settings.MODEL_TYPE == 1:
            model = model_class.frcnn_pretrained_model()

        model.to(self.device)

        # Optimizer
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(parameters, lr=learning_rate)  # Adam
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        # Initialization
        best = 0.0  # initial value of best metric
        train_loss_list = []
        validation_map_list = []

        t1 = time.time()

        for epoch in range(epochs):
            # One epoch train
            train_loss, average_train_loss = self.train_epoch(
                model, data_loader_train, optimizer, scheduler
            )
            print(
                'epoch{} training loss: {}, training average loss: {}'.format(
                    epoch, train_loss, average_train_loss
                )
            )
            scheduler.step(train_loss)

            # Validation
            map_mean = self.evaluation(model, data_loader_valid)

            print('epoch{} map_mean: {}'.format(epoch, map_mean))

            best_model = None
            if best <= map_mean:
                best = map_mean
                best_model = copy.deepcopy(model)

            train_loss_list.append(average_train_loss)
            validation_map_list.append(map_mean)

            # for hyper-parameter tuning
            tune.report(map=map_mean)

        t2 = time.time()

        print('Train loss: {:.3f}'.format(average_train_loss))
        print('Validation mAP: {:.3f}'.format(map_mean))
        print('Best mAP: {}'.format(best))

        print('Time: {}, {}'.format(t1 - t0, t2 - t1))

        path_trained_model = None
        if settings.IS_SAVE:
            path_trained_model = os.path.abspath(
                os.path.join(settings.PATH_MODELS, settings.MODEL_TRAIN_NAME + '.pth')
            )
            torch.save(best_model.state_dict(), path_trained_model)

        # MLflow
        with mlflow.start_run() as run:

            mlflow.log_param('batch_size', batch_size)
            mlflow.log_param('epochs', epochs)
            mlflow.log_param('learning_rate', learning_rate)

            mlflow.log_metric('mean_ap', best)
            mlflow.log_metric('train_loss', average_train_loss.item())

            mlflow.pytorch.log_model(best_model, 'model')

        return path_trained_model

    ### Inference for test dataset
    def inference(self, path_trained_model, path_dataset, config):

        # Load model
        model_class = models.Model(settings.NUM_CLASSES)
        if settings.MODEL_TYPE == 0:
            model_load = model_class.frcnn_vanilla_model()
        elif settings.MODEL_TYPE == 1:
            model_load = model_class.frcnn_pretrained_model()

        model_load.load_state_dict(
            torch.load(path_trained_model, map_location=self.device)
        )
        model_load.to(self.device)

        # Batch size
        batch_size = config['batch_size']

        # Load Data
        _, _, dataset_test = self.transform(path_dataset)
        data_loader_test = DataLoader(
            dataset_test,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=utils.collate_fn,
        )

        # Evaluation
        map_mean = self.evaluation(model_load, data_loader_test)
        print('Test mAP: {:.3f}'.format(map_mean))

        # Save model
        path_saved_model = None
        if settings.IS_SAVE:
            path_saved_model = os.path.abspath(
                os.path.join(settings.PATH_MODELS, settings.MODEL_SAVED_NAME + '.pth')
            )
            torch.save(model_load.state_dict(), path_saved_model)

        # MLflow
        with mlflow.start_run() as run:

            mlflow.log_metric('mean_ap', map_mean)

            mlflow.pytorch.log_model(model_load, 'model')

        return path_saved_model

    ### Hepyr-parameter tuning
    def hyper_parameter_tuning(self, path_data):
        # Initialize
        ray.init(
            runtime_env={"working_dir": os.path.abspath(settings.PATH_WORKING_DIR)},
        )

        # Heper-parameter
        hyper_parameters = {
            'batch_size': tune.choice(settings.HYPER_PARAMETER['batch_size']),
            'learning_rate': tune.choice(settings.HYPER_PARAMETER['learning_rate']),
            'epochs': tune.choice(settings.HYPER_PARAMETER['epochs']),
        }

        # Scheduler
        scheduler = tune.schedulers.ASHAScheduler(
            metric='map',
            mode='max',
            max_t=settings.MAX_EPOCHS,
            grace_period=1,
            reduction_factor=2,
        )

        # Tuning execution
        result = tune.run(
            partial(self.train, path_data=path_data),
            resources_per_trial={'cpu': 2},
            config=hyper_parameters,
            num_samples=1,
            scheduler=scheduler,
            stop={'training_iteration': 10},
        )

        print('tune result: {}'.format(result))
        best_params = result.get_best_config(metric='map', mode='max', scope='last')
        print('best params: {}'.format(best_params))
        return best_params
