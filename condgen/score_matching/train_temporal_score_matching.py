from distutils.util import strtobool
from argparse import ArgumentParser
from data_utils import PendulumDataModule, SyntheticMMDataModule
from cv_data_utils import CVDataModule
from models.score_matching import TemporalScoreMatcher
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def main(model_cls, data_cls, args):
    dataset = data_cls(**vars(args))
    dataset.prepare_data()
    conditional_dim = dataset.conditional_dim
    conditional_len = dataset.conditional_len
    output_dim = dataset.output_dim
    static_dim = dataset.baseline_size
    
    model = model_cls(conditional_dim = conditional_dim, output_dim = output_dim, conditional_len = conditional_len, static_dim = static_dim, **vars(args))

    logger = WandbLogger(
        name=f"Temporal-matcher",
        project="dmm",
        entity="edebrouwer",
        log_model=False
    )
    
    checkpoint_cb = ModelCheckpoint(
        dirpath=logger.experiment.dir,
        monitor='val_loss',
        mode='min',
        verbose=True
    )
    early_stopping_cb = EarlyStopping(monitor="val_loss", patience=10)

    trainer = pl.Trainer(gpus = 1, logger = logger, callbacks = [checkpoint_cb, early_stopping_cb], max_epochs = args.max_epochs)
    trainer.fit(model, datamodule = dataset)

    checkpoint_path = checkpoint_cb.best_model_path
    trainer2 = pl.Trainer(logger=False)

    model = model_cls.load_from_checkpoint(
        checkpoint_path)
    val_results = trainer2.test(
        model,
        test_dataloaders=dataset.val_dataloader()
    )[0]

    val_results = {
        name.replace('test', 'val'): value
        for name, value in val_results.items()
    }

    test_results = trainer2.test(
        model,
        test_dataloaders=dataset.test_dataloader()
    )[0]

    for name, value in {**test_results}.items():
        logger.experiment.summary['restored_' + name] = value
    for name, value in {**val_results}.items():
        logger.experiment.summary['restored_' + name] = value

if __name__=="__main__":
    
    parser = ArgumentParser()

    # figure out which model to use and other basic params
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--data_type', type = str, default = "MMSynthetic")
    parser.add_argument('--model_type', type = str, default = "ScoreMatching")
    parser.add_argument('--fold', default=0, type=int, help=' fold number to use')

    partial_args, _ = parser.parse_known_args()

    model_cls = TemporalScoreMatcher
    
    if partial_args.data_type == "MMSynthetic":
        data_cls = SyntheticMMDataModule
    elif partial_args.data_type == "Pendulum":
        data_cls = PendulumDataModule
    elif partial_args.data_type == "CV":
        data_cls = CVDataModule
    parser = model_cls.add_model_specific_args(parser)
    parser = data_cls.add_dataset_specific_args(parser)
    args = parser.parse_args()

    main(model_cls, data_cls, args)
