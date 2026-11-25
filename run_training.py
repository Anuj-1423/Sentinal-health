"""Train and evaluate the Sentinel Health CKD classifier."""
from pathlib import Path
import argparse, json
from sentinel_health.ckd_data import DATA_PATH
from sentinel_health.ckd_training import ARTIFACTS, run_training

def main():
    parser=argparse.ArgumentParser(description="Run 30 CKD model/preprocessing configurations with stratified validation.")
    parser.add_argument("--data",default=str(DATA_PATH))
    parser.add_argument("--artifacts",default=str(ARTIFACTS))
    parser.add_argument("--cv-folds",type=int,default=5)
    args=parser.parse_args()
    result=run_training(args.data,args.artifacts,cv_folds=args.cv_folds)
    report=result["report"]
    print("Selected model:",report["selected_model"])
    print("Holdout metrics:",json.dumps(report["test_metrics"],indent=2))
    print("Report:",Path(args.artifacts).resolve()/"training_report.json")

if __name__=="__main__":main()
