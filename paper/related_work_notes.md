# Related Work Evidence Notes

These notes are inputs to the manuscript, not manuscript prose. Verify final
bibliographic metadata before submission.

## PatchCore and MVTec AD

- Bergmann et al. introduced MVTec AD with 5,354 high-resolution images across
  object and texture categories, test anomalies and pixel-precise masks. This
  supports the dataset and metric description.
- Roth et al. introduced PatchCore's nominal patch-feature memory bank and greedy
  coreset subsampling. The original paper explicitly motivates coreset reduction
  as a storage and inference-time mechanism, making our on-device coreset study a
  deployment analysis rather than a new anomaly detector.
- Anomalib supplies the training/evaluation implementation; our contribution is
  the traced TensorRT/C++/CUDA runtime and deployment trade-off study.

Primary sources:

- https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html
- https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html
- https://github.com/open-edge-platform/anomalib

## Edge anomaly detection

- PaSTe (CVPRW 2025) directly addresses visual anomaly detection efficiency at
  the edge through lightweight backbones and teacher/student sharing. It is a
  close contemporary comparison, but its main lever differs from our unchanged
  WideResNet50-2 feature representation and memory-bank/runtime optimization.
- Rolih et al. study high-resolution memory efficiency through tiled ensembles.
  Their problem is input/model memory scaling; ours is exact memory-bank search
  and end-to-end embedded execution.

Primary sources:

- https://openaccess.thecvf.com/content/CVPR2025W/VAND/papers/Barusco_PaSTe_Improving_the_Efficiency_of_Visual_Anomaly_Detection_at_the_CVPRW_2025_paper.pdf
- https://openaccess.thecvf.com/content/CVPR2024W/VAND/html/Rolih_Divide_and_Conquer_High-Resolution_Industrial_Anomaly_Detection_via_Memory_Efficient_CVPRW_2024_paper.html

## GPU retrieval and runtime

- Johnson et al. establish GPU similarity-search design as a strong comparison
  point. A maintained Faiss GPU comparison should be attempted if the Jetson
  software stack supports it; inability to install it must be documented rather
  than replaced with an unsupported claim.
- TensorRT documentation supports the precision discussion: enabling FP16 or
  INT8 permits lower-precision tactics but does not guarantee that every layer
  executes at that precision.
- CUDA documentation motivates pinned memory, streams and asynchronous global-
  to-shared copies. Our performance claims still come from measured ablations,
  not from documentation expectations.

Primary sources:

- https://github.com/facebookresearch/faiss
- https://docs.nvidia.com/deeplearning/tensorrt/latest/
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/
