# 原研究图片

本目录保存原研究的展示图片。以下四张组合图已加入主 README，按“图像指标提取 → SHAP 解释 → 空间应用”排列；依赖图默认折叠。图片按用户提供的原文件复制，没有裁剪、重绘或改变数值。

| 已加入图片 | 内容 | 主 README 展示 |
| --- | --- | --- |
| [segmentation-overview.png](segmentation-overview.png) | 街景原图、语义 mask、叠加图，以及全景/鱼眼指标流程 | 完整展示 |
| [shap-global-summary.png](shap-global-summary.png) | SHAP 平均绝对重要性与贡献分布 | 完整展示 |
| [shap-dependence-panels.png](shap-dependence-panels.png) | GVI、生活服务 POI、TVF、SVF 的依赖图 | 可展开区域 |
| [renovation-spatial-maps.png](renovation-spatial-maps.png) | 改造需求分类和改造潜力空间分布 | 完整展示 |

流程组合图保留了原来的 DeepLab 标注；当前可运行的正式推理入口使用 SegFormer。图示不能替代模型配置或原 GIS 工程。

## 后续添加原始单图

把图片放在本目录，在主 README 对应位置加入引用，例如：

```markdown
![Original study SHAP feature importance](docs/historical-figures/shap_importance_explained.png)
```

只为实际已加入的文件添加图片引用。较长的补充图可以继续放在 `<details>` 折叠区；在 GitHub Desktop 中将图片和 README 一起提交并推送。

## 标题与文件名

| 原始单图类型 | 建议文件名 | 其他已找到的原版本文件名 |
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
