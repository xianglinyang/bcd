# Block Coordinate Decent

## Run Instruction
1. 在文件里设置hyperparameters
2. 直接执行
```console
$ bash scripts/algo1.sh
$ bash scripts/algo2.sh
```

## Plot Instruction
```console
# 单个run的plot
$ python src/plots/plot_one.py --csv runs/algo1/loss_time.csv --save_dir runs/algo1

# 多个run一起的
$ python plot_compare_seaborn.py \
  --csv1 runs/algo1_loss_time.csv --label1 "Algo1 (MBCD)" \
  --csv2 runs/algo2_loss_time.csv --label2 "Algo2 (L-MBCD, K=5)" \
  --save_dir runs --smooth 50
```

# Eval Instruction