import argparse

from deepwaters.utils import wandb_checkpoint_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project")
    parser.add_argument("run_id")
    parser.add_argument(
        "-a", "--alias", default="best", help="'best', 'latest', or 'v<int>'"
    )
    args = parser.parse_args()

    ckpt_file = wandb_checkpoint_download(
        project=args.project, run_id=args.run_id, alias=args.alias
    )
    print(f"Downloaded model checkpoint checkpoint to '{ckpt_file}'.")


if __name__ == "__main__":
    main()
