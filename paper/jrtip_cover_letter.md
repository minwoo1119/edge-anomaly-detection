# Cover Letter — Journal of Real-Time Image Processing

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
   ablations. Asynchronous tiled CUDA search provides 28.43× end-to-end speedup
   over CPU under a fixed data path.
3. Evaluation across all 15 MVTec AD categories, reaching macro image/pixel
   AUROC of 0.983314/0.979761 with category-specific memory banks.
4. A 4,000-frame, 33.94-minute sustained test that separates a 550 ms input
   cycle from a 1.1 s response deadline and reports p95 behavior, power,
   temperature, and a predefined performance-degradation criterion.

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
