# JRTIP Submission Checklist

Checked against the official JRTIP submission guidelines on 2026-10-13.

## Completed

- [x] Journal-specific two-column `svjour3` source uses the official macro
  package linked from the journal page.
- [x] Abstract is between 150 and 250 words.
- [x] Six indexing keywords are supplied.
- [x] Decimal headings use at most three levels.
- [x] Numeric citations and a numbered Springer bibliography style are used.
- [x] Figures are cited in order, use publisher-style names, and are supplied as
  vector EPS originals plus vector PDF compilation copies in a flat directory.
- [x] Tables are cited in order and include captions.
- [x] Accuracy, latency, p95, power, energy, memory, temperature, and sustained
  behavior are supported by the frozen evidence manifest.
- [x] Real-time issues are treated as input-cycle and response deadlines rather
  than processing time alone.
- [x] Data availability, code availability, and generative-AI-use statements
  are included.
- [x] A cover letter explains the specific real-time contribution.
- [x] Editable sources and official style files are collected in one flat ZIP.

## Required author metadata

- [x] Add Minwoo Lee and corresponding author HyeongTae Ahn in author order.
- [x] Set the running author line to `Lee and Ahn`.
- [x] Add both authors' department, institution, city, and country.
- [ ] Add the corresponding author's active email and optional 16-digit ORCID.
- [ ] Confirm the funding statement.
- [ ] Confirm the competing-interest statement.
- [ ] Add a CRediT contribution statement covering every author.
- [ ] Complete the bracketed fields in `cover_letter.md`.
- [ ] Confirm that the work is not published or under review elsewhere and that
  every co-author and responsible institution approves submission.

## Required technical revision before submission

- [x] Complete matched cuBLAS and available FAISS GPU exact-retrieval baselines.
- [ ] Complete the Nsight explanation with occupancy and achieved-bandwidth
  counters; the archived Systems traces already establish kernel composition,
  allocation overhead, and synchronization behavior.
- [ ] Run multi-category, multi-ratio, multi-seed validation-based selection and
  one locked held-out evaluation.
- [x] Implement and measure a native FP16 device memory bank with FP32 accumulation.
- [x] Run fixed-rate arrivals and report p99/max, deadline misses, backlog, drops,
  temperature, power, and sustained drift.
- [ ] Regenerate every table, figure, claim audit, PDF, and source archive from
  the revised frozen evidence set.

## Final production checks

- [x] Compile with the official class and inspect the PDF for a maximum of 12
  double-column pages including references and any author biography.
- [x] Check that PDF fonts are embedded and figure labels remain legible at 84 or
  174 mm width.
- [x] Confirm every reference is cited and every citation has a reference.
- [ ] Upload both the compiled PDF and `jrtip_submission_sources.zip`.
- [ ] Enter author contributions and competing interests in the submission
  interface; the journal states that interface entries control final publication.
- [ ] Archive the submitted PDF and source ZIP with their SHA-256 values.

Official instructions:
https://link.springer.com/journal/11554/submission-guidelines
