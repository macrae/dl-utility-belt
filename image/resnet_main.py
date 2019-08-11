import argparse


def run_job(args):

    # init args
    path = job_specs["path"]
    train = job_specs["train"]
    valid_pct = job_specs["valid_pct"]
    size = job_specs["size"]
    num_workers = job_specs["num_workers"]

    data = ImageDataBunch.from_folder(
        path=path,
        train=train,
        valid_pct=valid_pct,
        ds_tfms=get_transforms(),
        size=size,
        num_workers=num_workers,
    ).normalize(imagenet_stats)

    learn = cnn_learner(data, models.resnet34, metrics=error_rate)

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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        required=False,
        default=False,
        help="Default overwrite false",
    )

    args = parser.parse_args(main_args)

    job_specs = {i: args.__getattribute__(i) for i in args.__dir__() if i[0] != "_"}

    learn = run_job(job_specs)

    learn.save("stage-1")

    if args.overwrite:
        inner_join_df.write.mode("overwrite").save(args.target)
    else:
        inner_join_df.write.save(args.target)


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
