# Paper workspace

The authoritative manuscript source is `jrtip_manuscript.tex`. The current
working submission package is generated with:

```bash
python3 scripts/build_jrtip_submission.py \
  --template-zip /path/to/official-svjour3-template.zip \
  --output-dir results/submission_jrtip_<date>
```

`manuscript_draft.md` and `manuscript_outline.md` preserve earlier drafting
material and are not the submission source. Quantitative claims must agree with
`claims_and_evidence.md` and the referenced result artifacts.

The manuscript now includes the completed cuBLAS/FAISS exact-search comparison,
native FP16 GPU bank, and timestamped fixed-rate experiments. The remaining
technical gates before submission are recorded in
`jrtip_submission_checklist.md`: the full multi-category, multi-ratio,
multi-seed selection matrix with one locked held-out evaluation; Nsight Compute
occupancy/bandwidth counters; and final PDF/source regeneration after those
results are incorporated. Author-supplied email, funding, competing-interest,
and contribution statements also remain intentionally blank.
