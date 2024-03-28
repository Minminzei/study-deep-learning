from argparse import ArgumentParser
from original import Original
from resnet import ResNet
from efficientnet import EfficientNet


def summary(model: str, type: str = "fromzero"):
    if model == "original":
        Original().summary()
    elif model == "resnet":
        ResNet(type).summary()
    elif model == "efficientnet":
        EfficientNet(type).summary()
    else:
        print(f"サポートしていないモデルです: {model}")


def train(model: str, type: str = "fromzero"):
    if model == "original":
        Original().train()
    elif model == "resnet":
        ResNet(type).train()
    elif model == "efficientnet":
        EfficientNet(type).train()
    else:
        print(f"サポートしていないモデルです: {model}")


def predict(model: str, type: str = "fromzero"):
    if model == "original":
        Original().predict()
    elif model == "resnet":
        ResNet(type).predict()
    elif model == "efficientnet":
        EfficientNet(type).predict()
    else:
        print(f"サポートしていないモデルです: {model}")


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "function_name",
        type=str,
        help="実行するメソッド名{summary, train, predict}",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="学習モデル{original, resnet, efficientnet}",
        default="original",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        help="学習タイプ{fromzero, transfer_learning, fine_tuning}",
        default="fromzero",
    )
    return parser.parse_args()


def _main():
    args = get_args()
    if args.function_name == "summary":
        summary(args.model, args.type)

    elif args.function_name == "train":
        train(args.model, args.type)

    elif args.function_name == "predict":
        predict(args.model, args.type)


if __name__ == "__main__":
    _main()
