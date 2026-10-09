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
values. These results demonstrate why edge anomaly-detection studies must report
the complete retrieval pipeline and distinguish latency, throughput interval,
storage, and runtime memory. Multi-category and coreset results are pending and
will replace the marked placeholders before submission.

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
benchmarks. The final submission will add five independently trained coreset
ratios and all 15 MVTec AD categories.

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

The precision comparison fixes the 10% FP32 bank and final S5 retrieval path.
INT8 uses entropy calibration from 100 train-normal tensors. The system
comparison fixes the FP16 engine, bank, retrieval kernel, and executable while
varying S0–S6. FP16 bank storage is derived deterministically from the FP32
export, then restored to FP32 by the current runtime loader.

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

### 5.4 Pending final experiments

**Coreset ablation:** `[PLACEHOLDER: independently trained 1%, 2.5%, 5%, 10%,
20% results and Pareto analysis]`.

**Retrieval backend ablation:** `[PLACEHOLDER: controlled CPU, direct CUDA,
warp, tiled, asynchronous tiled, and transpose-assisted comparison]`.

**Multi-category evaluation:** `[PLACEHOLDER: all 15 MVTec AD category results,
macro averages, representative object/texture performance]`.

## 6. Discussion

The results separate three quantities that are often conflated. TensorRT engine
time describes only feature extraction. End-to-end single-frame latency includes
retrieval and postprocessing. Pipeline interval describes steady-state
throughput when frames overlap. Likewise, a smaller serialized bank does not
imply lower runtime memory when the loader expands its values.

The current system processes roughly two bottle images per second. A real-time
claim is valid only for an inspection process whose specified cycle and response
deadline are consistent with measured p95 interval and completion time. The
final paper will add a sustained thermal run and state the deadline explicitly.

The study currently has three material limits: evidence covers one category,
one embedded platform, and exact brute-force retrieval. The coreset and
multi-category experiments are required before claiming generality. Approximate
retrieval and alternate backbones are useful future comparisons but are outside
the fixed-representation systems question studied here.

## 7. Conclusion

On Jetson Orin Nano, PatchCore deployment is governed by the complete
memory-bank pipeline rather than the TensorRT engine in isolation. FP16 preserves
bottle accuracy and greatly accelerates feature extraction, yet exact retrieval
limits the end-to-end benefit. Pinned staging and controlled pipeline overlap
provide measurable systems gains, while entropy INT8 and FP16 bank serialization
show accuracy and runtime-memory limitations that must be reported explicitly.
The final claims will be frozen after the coreset, retrieval-backend,
multi-category, and sustained-run gates are complete.
