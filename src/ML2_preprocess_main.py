import argparse
import yaml
from ML2_preprocess_main import run_preprocessing
from ML2_Trainmodel import run_training
from ML2_Evaldrift import run_drift_evaluation

def main(config_path):
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Step 1: Preprocess data
    run_preprocessing(cfg["2011_path"], cfg["2012_path"], cfg.get("output_path"))

    # Step 2: Train model
    run_training(cfg.get("model_path", "best_model.pkl"))

    # Step 3: Evaluate drift
    run_drift_evaluation(cfg["2011_path"], cfg["2012_path"], cfg.get("model_path", "model.pkl"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
