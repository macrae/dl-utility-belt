import argparse
import sys

from fastai.vision import ImageDataBunch, cnn_learner


def train_model(hyperparams):
    """Train a resnet model...

    Parameters
    ----------
    hyperparams : dict
        dict of hyperparams

    Returns
    -------
    [type]
        trained pytorch model
    """

    # init args
    path = hyperparams["path"]
    train = hyperparams["train"]
    valid_pct = hyperparams["valid_pct"]
    size = hyperparams["size"]
    num_workers = hyperparams["num_workers"]

    # load data from folder
    data = ImageDataBunch.from_folder(
        path=path,
        train=train,
        valid_pct=valid_pct,
        ds_tfms=get_transforms(),
        size=size,
        num_workers=num_workers,
    ).normalize(imagenet_stats)

    # init model
    learn = cnn_learner(data, models.resnet34, metrics=error_rate)

    # train model
    learn.fit_one_cycle(4)

    return learn


def main(main_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="path to...")
    parser.add_argument("--classes", required=True, help="image classes to...")
    parser.add_argument("--train", required=True, help="train...")
    parser.add_argument("--valid_pct ", required=True, help="validation set percent")
    parser.add_argument("--size ", required=True, help="size of...")
    parser.add_argument("--num_workers ", required=True, help="number of workers...")
    parser.add_argument("--save_to ", required=True, help="path to save...")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        required=False,
        default=False,
        help="Default overwrite false",
    )

    # unpack args
    args = parser.parse_args(main_args)
    hyperparams = {i: args.__getattribute__(i) for i in args.__dir__() if i[0] != "_"}

    # train learner
    learn = train_model(hyperparams)

    # save model artifacts
    learn.save(args.save_to)

    if args.overwrite:
        learn.save(args.save_to)  # overwrite
    else:
        learn.save(args.save_to)  # do not overwrite


if __name__ == "__main__":
    sys.exit(main(sys.argv))
