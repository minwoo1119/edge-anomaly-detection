# End-to-End Optimization of Memory-Bank Anomaly Detection on NVIDIA Jetson Orin Nano

## Abstract

Memory-bank anomaly detectors can retain strong unsupervised detection accuracy,
but their deployment cost is easily understated when evaluation reports only
the neural feature extractor. We implement PatchCore as a traced C++17,
TensorRT, and CUDA runtime on an NVIDIA Jetson Orin Nano and measure
preprocessing, transfer, feature extraction, exact nearest-neighbor retrieval,
postprocessing, power, and energy. On the MVTec AD bottle category, replacing
an FP32 TensorRT engine with FP16 reduces feature-extractor latency from 20.212
to 3.785 ms while reducing end-to-end latency only from 558.511 to 550.245 ms,
because exact memory-bank search remains the dominant stage. A controlled
S0–S6 systems ablation shows that pinned host staging provides the largest
single-frame data-path improvement and that a two-frame pipeline reaches a
482.350 ms steady-state interval at 6.426 J/image while preserving image and
pixel AUROC of 1.0 and 0.985455. Entropy-calibrated INT8 is an unfavorable
trade-off in this setting: image AUROC falls to 0.688889 for a 4.9% latency
reduction relative to FP16. FP16 bank serialization halves the artifact payload
without reducing runtime memory because the current loader restores FP32
values. Across all 15 MVTec AD categories, the selected FP16 runtime reaches
macro image and pixel AUROC of 0.983314 and 0.979761; category mean latency
ranges from 164.050 to 906.637 ms as memory-bank size changes. These results
demonstrate why edge anomaly-detection studies must report the complete
retrieval pipeline and distinguish latency, throughput interval, storage, and
runtime memory. A five-ratio bottle coreset study further shows that reducing
the bank from 20% to 1% lowers mean latency from 992.520 to 72.093 ms and bank
size from 250.799 to 12.539 MB while retaining image AUROC 1.0.
Under a fixed S0 data path, asynchronous tiled CUDA retrieval reduces bottle
exact-search latency by 35.70x and end-to-end latency by 28.43x relative to CPU.

**Keywords:** industrial anomaly detection; PatchCore; edge inference; TensorRT;
CUDA; nearest-neighbor search; Jetson Orin Nano

## 1. Introduction

Industrial inspection systems often have many normal samples but few stable
examples of every possible defect. Unsupervised anomaly detection addresses this
setting by learning normality and treating deviations as potential defects.
MVTec AD provides a standard evaluation benchmark with object and texture
categories, image-level labels, and pixel-level masks [@bergmann2019mvtec].
PatchCore obtains strong accuracy by storing nominal patch embeddings and
matching test patches against a coreset of this memory bank [@roth2022patchcore].

The same design creates an edge-deployment challenge. A TensorRT feature
extractor can execute in a few milliseconds while exact patch-to-bank search
takes hundreds or thousands of milliseconds. Reporting engine latency alone
therefore does not describe application throughput, energy, or deadline
behavior. Data layout and transfer choices also determine whether GPU retrieval
is efficient: the TensorRT output is channel-major NCHW, whereas distance
calculation benefits from contiguous patch-major query vectors.

This work studies the complete PatchCore deployment path on a Jetson Orin Nano.
The current evidence supports three contributions. First, we provide a
reproducible runtime that records stage latency, power, artifact lineage, and
accuracy using the same executable and model hashes. Second, we evaluate
persistent buffers, pinned staging, asynchronous copies, a dedicated CUDA
stream, GPU-resident retrieval, and two-frame overlap under controlled
conditions. Third, we quantify precision and storage trade-offs, including
negative INT8 and FP16-bank results that would be hidden by feature-extractor
benchmarks. Finally, we evaluate five independently trained bottle coreset
ratios and the selected configuration on all 15 MVTec AD categories using
independently trained category-specific memory banks.

## 2. Related Work

PatchCore represents nominal training images using intermediate pretrained CNN
features and selects a greedy coreset to reduce storage and search cost
[@roth2022patchcore]. We retain this anomaly detector and its WideResNet50-2
features; the contribution is the embedded runtime and its measured systems
trade-offs. Anomalib provides a reproducible anomaly-detection implementation
and evaluation framework [@akcay2022anomalib].

Recent work considers edge-efficient anomaly detection through model changes.
PaSTe uses lightweight representations and shared teacher/student computation
[@barusco2025paste]. Tiled ensembles address memory pressure for high-resolution
industrial inputs [@rolih2024tiled]. Our study instead holds the learned feature
representation fixed and isolates the cost of exact memory-bank retrieval and
the surrounding data path.

GPU similarity search is well established at larger scales, including the
Faiss GPU design [@johnson2021faissgpu]. Embedded unified-memory systems impose
different capacity, power, and integration constraints. CUDA streams, pinned
memory, and asynchronous global-to-shared transfers provide relevant mechanisms
[@nvidia2026cuda], while TensorRT supplies lower-precision feature-extractor
tactics [@nvidia2026tensorrt]. We treat documentation as implementation guidance;
all performance conclusions come from device measurements.

## 3. Runtime Architecture

The training pipeline uses only normal training images. A WideResNet50-2
extractor combines layer2 and layer3 features into a `1536 x 32 x 32` embedding
for a `256 x 256` RGB input. For bottle at a 10% coreset ratio, the memory bank
contains 21,401 vectors of dimension 1,536. The deployment runtime performs
image preprocessing, TensorRT inference, embedding layout conversion, exact
squared-L2 nearest-neighbor search, PatchCore score reweighting, and Gaussian
anomaly-map postprocessing.

The optimized CUDA path transposes TensorRT's device-resident NCHW output into
patch-major query vectors. A tiled kernel reuses bank rows in shared memory and
uses asynchronous copies where supported. Distances accumulate in FP32, and
ties select the lowest bank index. Only the small nearest-neighbor outputs and
the maximum-distance patch needed for score reweighting return to the host.

The system stages are: S0, transient device buffers and pageable synchronous
copies; S1, persistent device buffers; S2, persistent pinned host staging; S3,
asynchronous copies on the default stream; S4, a dedicated non-blocking stream;
S5, a GPU-resident TensorRT-output-to-retrieval path; and S6, overlap of next
frame preprocessing with the current runtime. S6 has two relevant quantities:
single-frame completion time and steady-state pipeline interval.

Every engine, memory bank, export manifest, configuration, executable, and
evaluation dataset receives a SHA-256 identifier. Accuracy and performance rows
are merged only when executable, configuration, engine, and memory-bank hashes
match. Threshold selection on the MVTec test set is prohibited.

## 4. Experimental Setup

Experiments run on a Jetson Orin Nano in `MAXN_SUPER_ID2`, with JetPack 6.2.1,
L4T 36.4.7, CUDA 12.6, TensorRT 10.3, and OpenCV 4.8. The runtime is compiled
in Release mode for compute capability 8.7. Accuracy uses all 83 bottle test
images and reports image and pixel AUROC. Performance uses one fixed normal
image, 50 warm-up iterations, 200 measured iterations, and three independent
runs. We report mean stage time, mean per-run p95, between-run standard
deviation, average and peak power, and energy per image. `tegrastats` is sampled
every 100 ms. No accuracy threshold is tuned on the test split.

The multi-category experiment evaluates all 15 MVTec AD categories. Each
category uses its own seed-42 model and independently selected 10% coreset. Full
test splits supply image and pixel AUROC, while three independent performance
runs use one fixed normal image per category under the same 50/200 protocol.
Macro averages give every category equal weight.

The precision comparison fixes the 10% FP32 bank and final S5 retrieval path.
INT8 uses entropy calibration from 100 train-normal tensors. The system
comparison fixes the FP16 engine, bank, retrieval kernel, and executable while
varying S0–S6. FP16 bank storage is derived deterministically from the FP32
export, then restored to FP32 by the current runtime loader.

The retrieval-backend comparison fixes the FP16 engine, 10% FP32 bank, S0 data
path, executable, and power mode. It varies only CPU, direct CUDA, warp, tiled,
asynchronous tiled, and transpose-assisted asynchronous tiled exact search.
This separates kernel organization from the S5 GPU-resident data path.

Sustained validation uses the final S6 runtime for 4,000 measured frames. We
predefine a greater than 5% increase from the first to last complete five-minute
mean interval as evidence of operational performance degradation. The evaluated
soft real-time envelope is a 550 ms input cycle and 1.1 s response deadline.

## 5. Results

### 5.1 Precision

FP32 and FP16 both achieve image AUROC 1.0; pixel AUROC changes from 0.985461
to 0.985455. FP16 reduces TensorRT time by 81.3%, from 20.212 to 3.785 ms, but
end-to-end latency falls only 1.48%, from 558.511 to 550.245 ms. Exact retrieval
still takes 467.057 ms in the FP16 runs. Feature-extractor-only reporting would
therefore exaggerate the application speedup.

INT8 reduces mean TensorRT time to 2.703 ms and total time to 523.076 ms, but
image and pixel AUROC fall to 0.688889 and 0.761489. We reject this calibration
path as a deployment configuration. The result is limited to entropy PTQ with
the recorded train-normal calibration set and does not cover QAT or selective
mixed precision.

### 5.2 System ablation

All S0–S6 configurations preserve image/pixel AUROC of 1.0/0.985455. Persistent
device buffers alone have little effect: S0 and S1 measure 548.333 and 547.476
ms. Pinned staging at S2 reduces mean single-frame time to 502.427 ms and also
stabilizes the runs. S3 measures 501.316 ms. S4 and S5 do not improve on S3
under the fixed retrieval kernel, measuring 508.594 and 503.152 ms; these
negative results show that individual CUDA mechanisms do not guarantee a
system-level gain.

S6 reaches a steady-state interval of 482.350 ms/image (2.073 images/s) and
6.426 J/image, compared with 548.333 ms/image and 7.121 J/image at S0. Its
single-frame completion measurement is 964.625 ms because two frames are in
flight. We therefore describe S6 as a throughput optimization and do not report
482.350 ms as response latency.

### 5.3 Memory-bank storage

FP16 serialization reduces the bank payload from 131,487,872 to 65,743,872
bytes while preserving image/pixel AUROC at 1.0/0.985455. Runtime bank memory
remains 125.396 MB for both files because FP16 values are restored to FP32.
Mean latency is 503.152 ms for the FP32 file and 523.113 ms for FP16, with large
NN run variation; no speedup is supported. Native FP16 device storage with FP32
accumulation remains a separate implementation question.

### 5.4 Coreset trade-off

The independently trained 1%, 2.5%, 5%, 10%, and 20% bottle coresets contain
2,140, 5,350, 10,700, 21,401, and 42,803 vectors. All five achieve image AUROC
1.0. Pixel AUROC changes only from 0.984961 at 1% to 0.985500 at 20%, a range of
0.000540.

System cost changes substantially. Mean end-to-end latency is 72.093, 155.172,
283.882, 570.156, and 992.520 ms as the ratio increases. Serialized bank size
is 12.539, 31.348, 62.695, 125.396, and 250.799 MB, and mean energy rises from
0.865 J/image at 1% to 13.599 J/image at 20%. Under image AUROC, the 1% variant
dominates every larger tested ratio. The result is specific to bottle and seed
42; the small pixel-level differences and independently trained lineage are
retained rather than treating the banks as truncations of one export.

### 5.5 Retrieval backend ablation

All six S0 backends preserve image/pixel AUROC of 1.0/0.985452. CPU exact
search takes 14434.802 ms and produces a 14505.975 ms end-to-end time. Direct
CUDA reduces these values to 4314.865 and 4420.934 ms. Warp reduction lowers
total latency to 844.294 ms, tiling to 613.777 ms, and asynchronous tiling to
510.285 ms. The best S0 backend therefore provides 35.70x NN and 28.43x
end-to-end speedup over CPU, while energy falls from 94.681 to 7.367 J/image.

Transpose-assisted asynchronous tiling measures 527.955 ms, 3.46% slower than
asynchronous tiling in S0. This negative result is consistent with the S0 host
layout conversion: the transpose specialization is intended for the S5
device-resident NCHW path and does not improve this controlled host-mediated
comparison. We retain both results and evaluate the data-path change separately
in the system ablation.

### 5.6 Multi-category evaluation

The selected FP16/S5 runtime reaches macro image AUROC 0.983314 and macro pixel
AUROC 0.979761 across all 15 MVTec AD categories. The object-category averages
are 0.978531/0.984692, and the texture-category averages are
0.992882/0.969898. Toothbrush has the lowest image AUROC (0.919444), while wood
has the lowest pixel AUROC (0.929994); these results prevent an unqualified
near-perfect-accuracy claim.

Mean end-to-end latency varies from 164.050 ms for toothbrush to 906.637 ms for
hazelnut. Their banks contain 6,144 and 40,038 entries, respectively. Across
categories, macro mean latency is 574.334 ms, mean throughput is 1.991 images/s,
mean power is 14.043 W, and mean energy is 8.128 J/image. TensorRT averages only
a few milliseconds in every category, whereas exact NN retrieval ranges from
119.257 to 780.257 ms, confirming that retrieval remains the system bottleneck
beyond the bottle case.

### 5.7 Sustained thermal behavior

The final S6 runtime processes 4,000 frames over 2,036.670 seconds. Its mean
and p95 pipeline intervals are 509.168 and 517.845 ms, corresponding to 1.964
images/s. Mean and p95 frame completion times are 1,018.309 and 1,031.660 ms.
It therefore meets the evaluated 550 ms input cycle and 1.1 s response deadline.

The first complete five-minute window averages 514.355 ms and the last averages
507.128 ms, a -1.405% change that passes the predeclared 5% degradation bound.
Maximum GPU temperature is 63.906 C, with 14.014 W mean input power. These data
show no sustained performance degradation during the run. Because the available
`tegrastats` format does not report a hardware throttle flag, this conclusion is
based on observed latency drift and temperature rather than a direct internal
throttling signal.

## 6. Discussion

The results separate three quantities that are often conflated. TensorRT engine
time describes only feature extraction. End-to-end single-frame latency includes
retrieval and postprocessing. Pipeline interval describes steady-state
throughput when frames overlap. Likewise, a smaller serialized bank does not
imply lower runtime memory when the loader expands its values.

The current system processes roughly two bottle images per second. Its measured
soft real-time envelope is one bottle input every 550 ms with a 1.1 s response
deadline. This wording ties the claim to observed p95 interval and completion
time and does not imply suitability for faster inspection lines.

The study currently has three material limits: it covers one benchmark, one
embedded platform, and exact brute-force retrieval. The coreset frontier is
demonstrated on bottle with a single seed and should not be assumed identical
for every category. Approximate retrieval and alternate backbones are useful
future comparisons but are outside the fixed-representation systems question
studied here.

## 7. Conclusion

On Jetson Orin Nano, PatchCore deployment is governed by the complete
memory-bank pipeline rather than the TensorRT engine in isolation. FP16 preserves
bottle accuracy and greatly accelerates feature extraction, yet exact retrieval
limits the end-to-end benefit. Pinned staging and controlled pipeline overlap
provide measurable systems gains, while entropy INT8 and FP16 bank serialization
show accuracy and runtime-memory limitations that must be reported explicitly.
The 15-category results show that the selected configuration retains strong
accuracy while retrieval remains dominant across diverse memory-bank sizes. The
coreset experiment shows that large edge-runtime gains are possible with a
small bank and negligible bottle accuracy change. A 33.94-minute sustained run
meets the stated 550 ms input cycle and 1.1 s response deadline without measured
performance degradation beyond the preregistered bound.
