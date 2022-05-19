from distutils.util import strtobool
from argparse import ArgumentParser
from data_utils.data_utils_MNIST import MNISTDataModule
from models.CFGAN import CFGAN

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def main(model_cls, data_cls, args):
    dataset = data_cls(**vars(args))
    dataset.prepare_data()
    #baseline_size = dataset.baseline_size
    #reconstruction_size = dataset.output_dim
    #input_long_size = dataset.conditional_dim
    #treatment_dim = dataset.treatment_dim
    #output_dims = dataset.output_dims
    #va_times = dataset.va_times
    #quantiles = dataset.quantiles
    #Y_surv_train = dataset.Y_surv_train
    #model = model_cls(baseline_size = baseline_size, input_long_size = input_long_size,  reconstruction_size = reconstruction_size, treatment_dim = treatment_dim, output_dims = output_dims, **vars(args))
    model = model_cls(**vars(args))
    
    logger = WandbLogger(
        name=f"CF_MNIST",
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

    trainer = pl.Trainer(gpus = args.gpus, logger = logger, callbacks = [checkpoint_cb, early_stopping_cb], max_epochs = args.max_epochs)
    trainer.fit(model, datamodule = dataset)

    checkpoint_path = checkpoint_cb.best_model_path
    trainer2 = pl.Trainer(logger=False, gpus = args.gpus)

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
    #parser.add_argument('--use_mask_train', type=strtobool, default=False, help='wether to use all patients in the training or only the ones who progress after T_mask')
    #parser.add_argument('--T_cond', default=0, type=int, help='T_condition')
    #parser.add_argument('--T_mask', default=-1, type=int, help='T_mask')
    parser.add_argument('--fold', default=0, type=int, help=' fold number to use')
    parser.add_argument('--gpus', default=1, type=int, help='the number of gpus to use to train the model')
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--max_epochs', default=500, type=int)
    parser.add_argument('--data_path', type=str, \
            default="/home/edward/Projects/MIT/ief/ief_core/checkpoints/")
    parser.add_argument('--model_type', type = str, default = "RNN")
    parser.add_argument('--data_type', type = str, default = "MMSynthetic")

    partial_args, _ = parser.parse_known_args()

    model_cls = CFGAN
    #if partial_args.model_type == "RNN":
    #    model_cls = RNN_seq2seq
    #elif partial_args.model_type == "NeuralODE":
    #    model_cls = NeuralODE
    #elif partial_args.model_type == "Transformer":
    #    model_cls = Transformer
    
    data_cls = MNISTDataModule

    #if partial_args.data_type == "MMSynthetic":
    #    data_cls = SyntheticMMDataModule
    #elif partial_args.data_type == "Pendulum":
    #    data_cls = PendulumDataModule
    #elif partial_args.data_type == "CV":
    #    data_cls = CVDataModule

    parser = model_cls.add_model_specific_args(parser)
    parser = data_cls.add_dataset_specific_args(parser)
    args = parser.parse_args()

    #if args.T_mask == -1:
    #    args.T_mask = args.T_cond

    main(model_cls, data_cls, args)
