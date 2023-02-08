from distutils.util import strtobool
from argparse import ArgumentParser
from condgen.data_utils.data_utils_MNIST import MNISTDataModule
from condgen.data_utils.data_utils_cf_traj import SimpleTrajDataModule
from condgen.models.CFGAN import CFGAN

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import copy
import os
import torch


def main(model_cls, data_cls, args):
    dataset = data_cls(**vars(args))
    dataset.prepare_data()

    # store the num_classes_model for later
    num_classes_model_og = copy.copy(args.num_classes_model)
    args.num_classes_model = 1
    model = model_cls(**vars(args))
    # model.set_classes(num_classes_model=1) #For pretraining, only a single model

    logger = WandbLogger(
        name=f"CF_{args.data_type}",
        project=f"counterfactuals",
        entity="edebrouwer",
        log_model=False
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(logger.experiment.dir, "stage1"),
        monitor='val_loss',
        mode='min',
        verbose=True
    )
    early_stopping_cb = EarlyStopping(
        monitor="val_loss", patience=args.early_stopping1)

    trainer = pl.Trainer(gpus=args.gpus, logger=logger, callbacks=[
                         checkpoint_cb, early_stopping_cb], max_epochs=args.max_epochs1)
    trainer.fit(model, datamodule=dataset)

    checkpoint_path = checkpoint_cb.best_model_path
    # trainer2 = pl.Trainer(logger=False, gpus = args.gpus)

    model = model_cls.load_from_checkpoint(
        checkpoint_path)

    preds = trainer.predict(model, dataset.train_dataloader())

    imgs_pred = torch.cat([pred["imgs_pred"]
                          for pred in preds]).detach().cpu()
    imgs_true = torch.cat([pred["imgs_true"] for pred in preds]).detach().cpu()
    imgs_x = torch.cat([pred["imgs_x"] for pred in preds]).detach().cpu()
    color = torch.cat([pred["color"] for pred in preds]).detach().cpu()
    noise_pred = torch.cat([pred["noise_pred"]
                           for pred in preds]).detach().cpu()

    from sklearn.cluster import MiniBatchKMeans
    import numpy as np
    from sklearn.metrics import confusion_matrix

    if num_classes_model_og == -1:
        total_clusters = args.num_classes
    else:
        total_clusters = num_classes_model_og
    # Initialize the K-Means model
    # Fitting the model to training set

    if args.continuous_relaxation:
        X_flat = noise_pred.reshape(imgs_pred.shape[0], 1)
    else:

        X_flat = (imgs_pred-imgs_true).reshape(imgs_pred.shape[0], -1)

    if args.cluster_algo == "kmeans":
        entropy_max = 0
        kmeans_max = None
        for kmean_attempt in range(10):
            kmeans = MiniBatchKMeans(n_clusters=total_clusters)
            kmeans.fit(X_flat)
            counts = np.unique(kmeans.labels_, return_counts=True)[1]
            ps = counts / counts.sum()
            entropy = -(np.log(ps)*ps).sum()
            if entropy >= entropy_max:
                kmeans_max = kmeans
                entropy_max = entropy

            confusion_mat = confusion_matrix(color, kmeans.labels_)
        # import ipdb; ipdb.set_trace()
    else:
        from sklearn.mixture import GaussianMixture
        entropy_max = 0
        kmeans_max = None
        for kmean_attempt in range(10):
            gmm = GaussianMixture(
                n_components=total_clusters, covariance_type="diag")
            gmm.fit(X_flat[np.random.choice(len(X_flat), 1000)])
            gmm_preds = gmm.predict(X_flat)
            counts = np.unique(gmm_preds, return_counts=True)[1]
            ps = counts / counts.sum()
            entropy = -(np.log(ps)*ps).sum()
            if entropy >= entropy_max:
                kmeans_max = gmm
                entropy_max = entropy

    if args.continuous_relaxation:
        args.num_classes_model = num_classes_model_og
        model = model_cls(**vars(args))
    else:
        model.reset_generator(
            num_classes_model=num_classes_model_og, num_classes=args.num_classes)

    model.set_kmeans(kmeans_max)

    checkpoint_cb2 = ModelCheckpoint(
        dirpath=os.path.join(logger.experiment.dir, "stage2"),
        monitor='val_loss',
        mode='min',
        verbose=True
    )

    early_stopping_cb2 = EarlyStopping(
        monitor="val_loss", patience=args.early_stopping2)

    trainer = pl.Trainer(gpus=args.gpus, logger=logger, callbacks=[
                         checkpoint_cb2, early_stopping_cb2], max_epochs=args.max_epochs2)
    trainer.fit(model, datamodule=dataset)
    trainer.test(model, dataloaders=dataset.val_dataloader())

    checkpoint_path = checkpoint_cb2.best_model_path

    model2 = model_cls.load_from_checkpoint(
        checkpoint_path, num_classes_model=num_classes_model_og)

    trainer2 = pl.Trainer(logger=False, gpus=args.gpus)

    val_results = trainer2.test(
        model2,
        dataloaders=dataset.val_dataloader()
    )[0]

    val_results = {
        name.replace('test', 'val'): value
        for name, value in val_results.items()
    }

    test_results = trainer2.test(
        model2,
        dataloaders=dataset.test_dataloader()
    )[0]

    for name, value in {**test_results}.items():
        logger.experiment.summary['restored_' + name] = value
    for name, value in {**val_results}.items():
        logger.experiment.summary['restored_' + name] = value


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--fold', default=0, type=int,
                        help=' fold number to use')
    parser.add_argument('--gpus', default=1, type=int,
                        help='the number of gpus to use to train the model')
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--repeat', default=1, type=int,
                        help="dummy argument for running multiple times")
    parser.add_argument('--max_epochs1', default=500, type=int)
    parser.add_argument('--max_epochs2', default=500, type=int)
    parser.add_argument('--early_stopping1', default=20, type=int)
    parser.add_argument('--early_stopping2', default=20, type=int)
    parser.add_argument('--data_path', type=str,
                        default="/home/edward/Projects/MIT/ief/ief_core/checkpoints/")
    parser.add_argument('--model_type', type=str, default="CFGAN")
    parser.add_argument('--data_type', type=str, default="MNIST")
    parser.add_argument('--cluster_algo', type=str, default="kmeans")

    partial_args, _ = parser.parse_known_args()

    model_cls = CFGAN
    if partial_args.data_type == "MNIST":
        data_cls = MNISTDataModule
    elif partial_args.data_type == "SimpleTraj":
        data_cls = SimpleTrajDataModule
    elif partial_args.data_type == "CV":
        data_cls = SimpleTrajDataModule

    parser = model_cls.add_model_specific_args(parser)
    parser = data_cls.add_dataset_specific_args(parser)
    args = parser.parse_args()

    main(model_cls, data_cls, args)
