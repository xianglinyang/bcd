# Block Coordinate Decent

## APIKEY
```console
$ nano ~/.bashrc
# 加入一行
export OPENROUTER_API_KEY="xxx"
# source ~/.bashrc
```

## Run Instruction
1. 根据自己的machine，修改`src/fsdp_config/llama_fsdp_config.yaml`
  - 主要是num_process=gpu的数量
  - 如果不是llamaLayer，需要修改
2. 在sh文件里设置hyperparameters, `scripts/algo1.sh`。
  - 主要fsdp config
  - 修改 hyperparameters
3. 执行
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

## Eval Instruction
需要把自己需要的hyperparameter在这个文件里定义。
结果存在`eval_results/common_utility.json`里.
```console
bash scripts/evaluate_downstream.sh
```