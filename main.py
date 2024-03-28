from argparse import ArgumentParser
from original import Original
from resnet import ResNet
from efficientnet import EfficientNet


def summary(model: str):
    if model == "original":
        Original().summary()
    elif model == "resnet":
        ResNet().summary()
    elif model == "efficientnet":
        EfficientNet().summary()
    else:
        print(f"サポートしていないモデルです: {model}")


def train(model: str):
    if model == "original":
        Original().train()
    elif model == "resnet":
        ResNet().train()
    elif model == "efficientnet":
        EfficientNet().train()
    else:
        print(f"サポートしていないモデルです: {model}")


def predict(model: str):
    if model == "original":
        Original().predict()
    elif model == "resnet":
        ResNet().predict()
    elif model == "efficientnet":
        EfficientNet().predict()
    else:
        print(f"サポートしていないモデルです: {model}")


def transfer_learning(model: str):
    if model == "efficientnet":
        EfficientNet("transfer_learning").train()
    else:
        print(f"転移学習で利用できるモデルはefficientnetのみです: {model}")


def fine_tuning(model: str):
    if model == "efficientnet":
        EfficientNet("fine_tuning").train()
    else:
        print(f"ファインチューニングで利用できるモデルはefficientnetのみです: {model}")


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "function_name",
        type=str,
        help="実行するメソッド名{summary, train, predict, transfer_learning, fine_tuning}",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="学習モデル{original, resnet, efficientnet}",
        default="original",
    )
    return parser.parse_args()


def _main():
    args = get_args()
    if args.function_name == "summary":
        summary(args.model)

    elif args.function_name == "train":
        train(args.model)

    elif args.function_name == "predict":
        predict(args.model)

    elif args.function_name == "transfer_learning":
        transfer_learning(args.model)

    elif args.function_name == "fine_tuning":
        fine_tuning(args.model)


if __name__ == "__main__":
    _main()
