import argparse
import youtokentome


def train(data_path, vocab_size, model_path):
    youtokentome.BPE.train(data=data_path, vocab_size=vocab_size, model=model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BPE training")
    parser.add_argument(
        "-d",
        "--data",
        default=None,
        type=str,
        help="data file path (required)",
        required=True
    )
    parser.add_argument(
        "-v",
        "--vocab",
        default=200,
        type=int,
        help="Vocabulary size (default: 200)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="BPE.model",
        type=str,
        help="Path to trained model (default: 'BPE.model')"
    )
    args = parser.parse_args()
    train(args.data, args.vocab, args.output)
