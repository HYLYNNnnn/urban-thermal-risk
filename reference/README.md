# Historical source snapshots

These files are retained unchanged for provenance, not as supported entry points. Run `segmentation/` and `analysis/` from the repository root instead. Importing some historical files can start processing or model downloads.

| Historical file | Actual behavior / limitation |
| --- | --- |
| `original_segmentation/Segment_deeplab_city.py` | Uses PyTorch/Hugging Face SegFormer-B5, despite the DeepLab filename |
| `original_segmentation/fisheyes.py` | Uses SegFormer-B1 on existing fisheye images; does not generate a fisheye projection |
| `original_segmentation/hw.py` | Empirical pixel-ratio H/W estimate and heuristic confidence, not a trained geometry model |
| `Segment_psnet_ade.py`, `Extract_Segment_psnet_ade.py`, `Segment_fcn_ade.py` | Legacy MXNet/GluonCV experiments using ADE20K models; need a separate old environment |
| `Segment_deeplab_ade.py`, `Segment_deeplab_voc.py` | VOC 21-class predictions padded/relabelled as ADE20K; category semantics do not match |
| `Segment_psnet_city.py` | VOC DeepLab predictions interpreted as Cityscapes labels; category semantics do not match |
| `Segment_fcn_coco.py` | Reads ADE20K labels for a different model category set |
| `original_analysis/main.py`, `original_analysis/explain.py` | Original THIR/XGBoost/SHAP versions, including the documented risk-boundary and evaluation issues |

The preserved `lable.csv` and ADE20K label description belong to the historical ADE experiments. They are never used to relabel the formal Cityscapes model. The formal adapter validates the checkpoint's actual configuration and 19 output channels.

The original sources came from the supplied archive and research folder. No model weights or full image batches are redistributed here. Model provenance: [NVIDIA B1](https://huggingface.co/nvidia/segformer-b1-finetuned-cityscapes-1024-1024), [NVIDIA B5](https://huggingface.co/nvidia/segformer-b5-finetuned-cityscapes-1024-1024). Their model cards link applicable model terms and upstream attribution.
