# 原研究图片放置说明

这里预留给你当时生成的 SHAP 和模型分析图片。主 README 已经写好标题、解释及对应的图片引用；当前图片尚未加入，因此引用先放在 HTML 注释中，不会显示损坏图片。

## 添加图片

1. 把原图复制到本文件所在的 `docs/historical-figures` 文件夹。
2. 按下表使用对应文件名；已有同名图片无需改名。
3. 打开仓库根目录 `README.md`，找到对应标题，删除那条图片引用前后的 `<!--` 和 `-->`。
4. 删除该图的 `Original figure pending: ...` 一行。标题和解释保留。
5. 在 GitHub Desktop 提交新增图片和 README 修改，再推送。

例如，放入 `shap_importance_explained.png` 后，把：

```markdown
<!-- ![Original study SHAP feature importance](docs/historical-figures/shap_importance_explained.png) -->
```

改为：

```markdown
![Original study SHAP feature importance](docs/historical-figures/shap_importance_explained.png)
```

也可以先只放图片，之后再让我帮你接上 README 中的图片引用。

## 标题与文件名

| 主 README 标题 | 预留文件名 | 其他已找到的原版本文件名 |
| --- | --- | --- |
| SHAP feature importance | `shap_importance_explained.png` | `shap_importance_bar_english.png`、`shap_importance_bar.png` |
| SHAP summary plot | `shap_summary_explained.png` | `shap_summary_plot_english.png`、`shap_summary_plot.png` |
| SHAP dependence plots | `shap_dependence_detailed.png` | 详细版多因素组合图 |
| Green view index dependence | `shap_dependence_gvi_english.png` | `shap_dependence_gvi.png` |
| Tree view factor dependence | `shap_dependence_tvf_english.png` | `shap_dependence_tvf.png` |
| Green view index and tree view factor interaction | `shap_interaction_explained.png` | `shap_interaction_gvi_tvf_english.png`、`shap_interaction_gvi_tvf.png` |
| Individual point explanation | `shap_individual_explained.png` | `shap_force_plot_example.png` 只有 force plot，图注需相应调整 |
| XGBoost training and evaluation | `xgboost_detailed_analysis.png` | 原训练诊断组合图 |
| Renovation priority analysis | `renovation_priority_matrix.png` | 原改造优先级矩阵 |

如果你想保留原文件名，可以直接修改主 README 中对应图片路径，不必把另一版图片伪装成同一张图。PNG、JPG 均可，路径中的扩展名需与实际文件一致，大小写也保持一致。

历史图片展示的是原来的实验结果；`docs/examples/` 是这次核查重新生成的输出。主 README 已分开说明这两部分，历史图中的成绩不应改写成本轮重新验证的成绩。

本文件夹没有被 `.gitignore` 排除，图片放进来后会出现在 GitHub Desktop 的改动列表中。
