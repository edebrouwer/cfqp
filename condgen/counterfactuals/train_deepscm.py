from distutils.util import strtobool
from argparse import ArgumentParser
from condgen.data_utils.data_utils_MNIST import MNISTDataModule
from condgen.data_utils.data_utils import PendulumDataModule
from condgen.models.deepscm import DeepSCM
from condgen.data_utils.data_utils_cf_traj import SimpleTrajDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def main(model_cls, data_cls, args):
    dataset = data_cls(**vars(args))
    dataset.prepare_data()
    
    input_dim = dataset.input_dim
    output_dim = dataset.output_dim
    conditional_len = dataset.conditional_len # length of the dimension along which we run the CNN (28 for images e.g.)
    model = model_cls(input_dim = input_dim, output_dim = output_dim, conditional_len = conditional_len, **vars(args))

    logger = WandbLogger(
        name=f"NF-CF",
        project="counterfactuals",
        entity="edebrouwer",
        log_model=False
    )
    
    checkpoint_cb = ModelCheckpoint(
        dirpath=logger.experiment.dir,
        monitor='val_loss',
        mode='min',
        verbose=True
    )
    early_stopping_cb = EarlyStopping(monitor="val_loss", patience=20)

    trainer = pl.Trainer(gpus = 1, logger = logger, callbacks = [checkpoint_cb, early_stopping_cb], max_epochs = args.max_epochs)
    trainer.fit(model, datamodule = dataset)

    checkpoint_path = checkpoint_cb.best_model_path
    trainer2 = pl.Trainer(logger=False, gpus = 1)

    model = model_cls.load_from_checkpoint(
        checkpoint_path)
    val_results = trainer2.test(
        model,
        dataloaders=dataset.val_dataloader()
    )[0]

    val_results = {
        name.replace('test', 'val'): value
        for name, value in val_results.items()
    }

    test_results = trainer2.test(
        model,
        dataloaders=dataset.test_dataloader()
    )[0]

    for name, value in {**test_results}.items():
        logger.experiment.summary['restored_' + name] = value
    for name, value in {**val_results}.items():
        logger.experiment.summary['restored_' + name] = value

if __name__=="__main__":
    
    parser = ArgumentParser()

    # figure out which model to use and other basic params
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--max_epochs', default=1000, type=int)
    parser.add_argument('--data_type', default="MNIST", type=str)
    parser.add_argument('--model_type', default="DeepSCM", type=str)

    partial_args, _ = parser.parse_known_args()

    model_cls = DeepSCM
    if partial_args.data_type == "MNIST":
        data_cls = MNISTDataModule
        parser.add_argument("--one_D",default = False)
    elif partial_args.data_type == "SimpleTraj":
        data_cls = SimpleTrajDataModule
        parser.add_argument("--one_D",default = True)

    #data_cls = PendulumDataModule
    parser = model_cls.add_model_specific_args(parser)
    parser = data_cls.add_dataset_specific_args(parser)
    args = parser.parse_args()
    main(model_cls, data_cls, args)
