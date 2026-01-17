# Block Coordinate Decent

## Run Instruction
1. 在文件里设置hyperparameters
2. 直接执行
```console
$ bash scripts/algo1.sh
$ bash scripts/algo2.sh
```

## Plot Instruction
跑完training之后，loss log会在`runs/`文件夹里，然后我们可以继续生成loss的图。
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
需要把自己需要的hyperparameter在这个文件里定义。
结果存在`eval_results/common_utility.json`里.
```console
bash scripts/evaluate_downstream.sh
```