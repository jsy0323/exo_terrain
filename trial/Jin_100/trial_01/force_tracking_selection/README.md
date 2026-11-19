# Jin_100 / trial_01 力跟踪周期筛选

两份原始记录各有 23 个完整周期。`trial_meta.json` 指向 `raw_exo_20261104_171304.csv`。

以 `valid_heel_strike_count` 增加的位置作为周期起点，用相邻两次有效足跟着地之间的区间划分完整周期（含起点，不含下一周期起点）。不纳入触发前的数据及末尾不完整周期。

默认按**整个周期**的 `sqrt(mean((measured_force - desired_force)^2))` 从小到大排序，每份记录选前 5 个周期。这样也计入期望力为零时的残余测量力。图中另列助力段（`desired_force >= 5 N`）的 RMSE。均按原始采样点计算，单位为 N。

横坐标用实测相邻足跟着地间隔归一化为 0–100%，不是控制器估计的 `cycle_phase_pct`；导出的样本表同时保留两种相位。末个样本通常略小于 100%，没有补点。曲线直接使用 CSV 中的 `desired_force` 和 `measured_force`，未额外滤波、平移、削峰或基线校正。

| 原始记录 | 入选周期（误差由小到大） | 最佳周期时间 / s | 最佳整周期 RMSE / N |
|---|---|---|---|
| 171304（元数据对应记录） | 1、3、22、17、6 | 1.427–2.684 | 15.87 |
| 171002 | 4、14、6、7、13 | 5.250–6.389 | 17.00 |

每份记录对应子目录内：

- `selected_cycles.png` / `.svg`：5 个入选周期及全部周期误差对比。
- `best_cycle.png` / `.svg`：最佳周期的相位图和时间图。
- `full_record_selection.png` / `.svg`：原始完整记录及入选区间。
- `cycle_ranking.csv`：全部完整周期的指标和排序。
- `selected_cycle_samples.csv`：入选周期原始力数据、时间和相位。

根目录的 `all_cycle_ranking.csv` 提供两份记录共 46 个周期的统一排序，`selection_summary.json` 记录筛选参数和来源。入选图呈现的是低误差周期，整体跟踪性能应结合全部周期指标判断。

在项目根目录重新生成：

```powershell
python trial\plot_force_tracking_cycles.py trial\Jin_100\trial_01
```

可用 `--top-n 3` 改为选取 3 个周期，或用 `--rank-by assistance` 改为按助力段误差排序；重新生成会覆盖该输出目录内同名分析文件。
