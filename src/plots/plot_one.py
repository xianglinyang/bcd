import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def maybe_smooth(df, y_col="loss", window=0):
    if window and window > 1:
        df = df.copy()
        df[f"{y_col}_smooth"] = df[y_col].rolling(window=window, min_periods=1).mean()
        return df, f"{y_col}_smooth"
    return df, y_col

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="runs/loss_time.csv")
    ap.add_argument("--save_dir", type=str, default="runs")
    ap.add_argument("--smooth", type=int, default=0, help="rolling mean window size, 0=off")
    args = ap.parse_args()

    df = pd.read_csv(args.csv).dropna()

    # seaborn style
    sns.set_theme(style="whitegrid", context="talk")

    # loss vs backward_calls
    df1, y1 = maybe_smooth(df, "loss", args.smooth)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df1, x="backward_calls", y=y1)
    plt.xlabel("backward_calls")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}/loss_vs_backward_seaborn.png", dpi=200)
    plt.close()

    # loss vs wall_time_sec
    df2, y2 = maybe_smooth(df, "loss", args.smooth)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df2, x="wall_time_sec", y=y2)
    plt.xlabel("wall_time_sec")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}/loss_vs_walltime_seaborn.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
