"""Train only the CIFAR-10 ResNet18 variant without shortcuts or BatchNorm."""
import os

from train_three import (
    build_model,
    inspect_model,
    make_data,
    parse_args,
    prepare_run,
    seed_all,
    train_one,
    write_csv,
)


MODEL_NAME = "resnet18_no_skip_no_bn"


def main():
    args = parse_args(default_out="./outputs_no_skip_no_bn")
    if args.inspect_only:
        seed_all(args.seed)
        print(MODEL_NAME)
        inspect_model(build_model(MODEL_NAME, args.dropout))
        return

    out = prepare_run(args)
    data = make_data(args, out)
    result = train_one(MODEL_NAME, args, data, out)
    write_csv(out / "metrics.csv", [result])


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
