import math
import os
from argparse import ArgumentParser
import pytorch_lightning as pl

from datamodule import DataModule
from model import SentimentClassifier, SWASentimentClassifier
from postprocess import postprocess


def get_callbacks(args):
    callbacks = [
        pl.callbacks.ModelCheckpoint(dirpath=args.output_path, every_n_epochs=1, save_on_train_epoch_end=False, monitor="val_acc", save_last=True, save_top_k=10, mode="max", auto_insert_metric_name=True)
    ]
    return callbacks

def console_args():
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="train or test")
    parser.add_argument("--model_type", type=str, default="xlnet")
    parser.add_argument("--model_name", type=str, default="hfl/chinese-xlnet-base", help="model name")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--gpus", type=int, default=1, help="number of gpus")

    parser.add_argument("--root_path", type=str, default="resources/nlp_data", help="root path")
    parser.add_argument("--train_data_path", type=str, default="resources/nlp_data/train.txt", help="train data path")
    parser.add_argument("--pseudo_data_path", type=str, help="pseudo data path", required=False)
    parser.add_argument("--test_data_path", type=str, default="resources/nlp_data/test.txt", help="test data path")
    parser.add_argument("--valid_size", type=int, default=2000, help="valid size")
    parser.add_argument("--num_workers", type=int, default=8, help="num workers")
    parser.add_argument("--train_batch_size", type=int, default=6, help="train batch size")
    parser.add_argument("--test_batch_size", type=int, default=32, help="test batch size")
    parser.add_argument("--max_length", type=int, default=900, help="max length")

    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--eps", type=float, default=1e-8, help="eps")
    parser.add_argument("--num_warmup_steps", type=int, help="warmup steps", required=False)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)

    parser.add_argument("--output_path", type=str, default="output", help="output path")
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--layer_norm", type=bool, default=False)
    parser.add_argument("--regression", type=bool, default=False)
    parser.add_argument("--r_drop", type=bool, default=False)
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--pooling_layers", type=int, default=1)
    parser.add_argument("--attack_epsilon", type=float, default=0.1)

    parser.add_argument("--gradient_clip_val", default=1.0, type=float)
    parser.add_argument("--gradient_clip_algorithm", default="norm", type=str)
    parser.add_argument("--accumulate_grad_batches", default=3, type=int)

    parser.add_argument("--max_epochs", type=int, default=10, help="epochs")
    parser.add_argument("--precision", type=int, default=16, help="precision")
    parser.add_argument("--seed", type=int, default=19260817, help="seed")
    parser.add_argument("--ckpt_path", type=str, required=False)
    parser.add_argument("--is_extra_output", type=bool, default=False)
    parser.add_argument("--use_swa", type=bool, default=False)
    parser.add_argument("--shuffle_valid", type=bool, default=True)
    parser.add_argument("--adv_train", type=bool, default=False)
    parser.add_argument("--optimize_f1", default=False, type=bool)

    return parser.parse_args()

if __name__ == "__main__":
    args = console_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    pl.seed_everything(args.seed)
    if args.adv_train:
        args.manual_gradient_clip_val = args.gradient_clip_val
        args.manual_gradient_clip_algorithm = args.gradient_clip_algorithm
        args.gradient_clip_val = args.gradient_clip_algorithm = None
    if args.mode == "train":
        args.num_training_steps = math.ceil(((len(open(args.train_data_path, "r+", encoding="utf-8").readlines()) - args.valid_size) / args.train_batch_size) * args.max_epochs / args.accumulate_grad_batches)
        if args.num_warmup_steps is None:
            args.num_warmup_steps = round(args.num_training_steps * args.warmup_proportion)
    datamodule = DataModule(**vars(args))
    if args.use_swa:
        model = SWASentimentClassifier(**vars(args))
    else:
        model = SentimentClassifier(**vars(args))
    model.datamodule = datamodule
    trainer: pl.Trainer = pl.Trainer.from_argparse_args(args, callbacks=get_callbacks(args), detect_anomaly=False)
    if args.mode == "train":
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=args.ckpt_path)
    elif args.mode == "test":
        if args.optimize_f1:
            trainer.validate(model=model, datamodule=datamodule, ckpt_path=args.ckpt_path)
        trainer.test(model=model, datamodule=datamodule, ckpt_path=args.ckpt_path)
        postprocess(prediction_file=os.path.join(args.output_path, "prediction.pkl"), raw_file=args.test_data_path, output_file=os.path.join(args.output_path, "section1.txt"))
    else:
        raise NotImplementedError