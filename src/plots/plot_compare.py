import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_tag(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path).dropna()
    df["algo"] = label
    return df

def add_smooth(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if window and window > 1:
        df = df.sort_values(["algo", "backward_calls"]).copy()
        df["loss_smooth"] = df.groupby("algo")["loss"].transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )
    else:
        df["loss_smooth"] = df["loss"]
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv1", type=str, required=True, help="Algo1 CSV")
    ap.add_argument("--label1", type=str, default="algo1")
    ap.add_argument("--csv2", type=str, required=True, help="Algo2 CSV")
    ap.add_argument("--label2", type=str, default="algo2")
    ap.add_argument("--save_dir", type=str, default="runs")
    ap.add_argument("--smooth", type=int, default=0, help="rolling mean window, 0=off")
    args = ap.parse_args()

    df = pd.concat([
        load_and_tag(args.csv1, args.label1),
        load_and_tag(args.csv2, args.label2),
    ], ignore_index=True)

    sns.set_theme(style="whitegrid", context="talk")
    df = add_smooth(df, args.smooth)

    # 1) vs backward_calls
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="backward_calls", y="loss_smooth", hue="algo")
    plt.xlabel("backward_calls")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}/loss_compare_vs_backward.png", dpi=200)
    plt.close()

    # 2) vs wall_time_sec
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="wall_time_sec", y="loss_smooth", hue="algo")
    plt.xlabel("wall_time_sec")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}/loss_compare_vs_walltime.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
