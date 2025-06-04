import argparse
import os

from train_model import train
from evaluate import evaluate
import app  

def main():
    parser = argparse.ArgumentParser(description="Community Comment Rating System")

    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "evaluate", "app"],
                        help="Mode to run: train / evaluate / app")

    args = parser.parse_args()

    if args.mode == "train":
        print("🚀 Starting Training...")
        train()

    elif args.mode == "evaluate":
        print("📊 Starting Evaluation...")
        evaluate()

    elif args.mode == "app":
        print("🌐 Launching Gradio App...")
        app.iface.launch()

if __name__ == "__main__":
    main()
