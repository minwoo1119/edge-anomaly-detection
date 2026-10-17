# Cover Letter — Journal of Real-Time Image Processing

> Working draft. Do not submit until the remaining multi-seed coreset experiment,
> manuscript regeneration, and author confirmations are complete.

Date: ____________________

Dear Editor-in-Chief,

Please consider our manuscript, “When TensorRT Is Not the Bottleneck:
End-to-End PatchCore Optimization on Jetson Orin Nano,” for publication as an
Original Research Article in the *Journal of Real-Time Image Processing*.

Minwoo Lee is an undergraduate student in the Department of Computer
Engineering at Kumoh National Institute of Technology. HyeongTae Ahn is a
professor in the same department and serves as the corresponding author.

The manuscript addresses a specific real-time image-processing problem:
memory-bank anomaly detectors can execute their neural feature extractor in a
few milliseconds while exact retrieval, data layout, transfers, and
postprocessing determine the actual input service interval and response time.
Reporting TensorRT latency alone therefore does not establish real-time
behavior. We implement and measure the complete PatchCore path on an NVIDIA
Jetson Orin Nano, distinguishing feature-extractor time, exact-search time,
single-frame completion latency, and steady-state pipeline interval.

The main contributions are:

1. A reproducible C++17/TensorRT/CUDA runtime with hash-linked accuracy,
   latency, power, energy, memory, and artifact provenance.
2. Controlled precision, coreset, exact-retrieval-backend, and S0–S6 system
   ablations. Under a controlled data path, cuBLAS exact search provides
   103.05× speedup over CPU and 3.73× over the best custom CUDA backend; matched
   FAISS GPU results reveal a bank-size-dependent backend choice.
3. Evaluation across all 15 MVTec AD categories, reaching macro image/pixel
   AUROC of 0.983314/0.979761 with category-specific memory banks, together with
   a native FP16 index that halves Bottle bank storage while preserving AUROC.
4. A timestamped bounded-queue fixed-rate evaluation. At 550-ms arrivals, a
   predeclared 30-minute run completes all 3,273 frames with zero drops or
   1.1-s deadline misses and 260.268/279.202-ms response p99/maximum; a 100-ms
   stress condition demonstrates measurable overload.

The work is original, has not been published previously, and is not under
consideration elsewhere. All authors have approved the manuscript and agree to
its submission.

Confirmation of the preceding statements: ________________________________

Competing-interest statement: ____________________________________________

Thank you for your consideration.

Sincerely,

HyeongTae Ahn, Professor and Corresponding Author<br>
Department of Computer Engineering<br>
Kumoh National Institute of Technology<br>
Gumi, Republic of Korea<br>
Postal address: ____________________<br>
Email: ____________________<br>
ORCID: ____________________
