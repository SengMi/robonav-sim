"""Run all experiments:  python main.py"""
from circuitsim.experiments import run_all

if __name__ == "__main__":
    for name, vals in run_all("results").items():
        print(f"\n[{name}]")
        for k, v in vals.items():
            print(f"  {k:24s} {v if v is None else round(v, 4)}")
    print("\nPlots saved in results/")
