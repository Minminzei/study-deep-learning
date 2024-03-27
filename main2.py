from argparse import ArgumentParser
from original import Original

parser = ArgumentParser()


class AlexNet:
    print("AlexNet")


def _main():
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
    args = parser.parse_args()
    if args.function_name == "summary":
        if args.model == "original":
            Original().summary()
        elif args.model == "resnet":
            # resnet().model.summary()
            print(">")
        elif args.model == "efficientnet":
            # efficientnet().model.summary()
            print(">")
        else:
            print("Invalid model name")
    elif args.function_name == "train":
        if args.model == "original":
            Original().train()
        elif args.model == "resnet":
            # train_resnet()
            print(">")
        elif args.model == "efficientnet":
            # train_efficientnet()
            print(">")
        else:
            print("Invalid model name")


if __name__ == "__main__":
    _main()
