#!/usr/bin/env python3
"""Audit the generated JRTIP package against the journal's mechanical rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path


REQUIRED_FILES = {
    "manuscript.tex", "references.bib", "svjour3.cls", "svglov3.clo",
    "spmpsci.bst",
    *(f"Fig{i}.{suffix}" for i in range(1, 7) for suffix in ("eps", "pdf")),
}


def extract(text: str, start: str, end: str) -> str:
    match = re.search(re.escape(start) + r"(.*?)" + re.escape(end), text, re.S)
    if not match:
        raise RuntimeError(f"missing block: {start} ... {end}")
    return match.group(1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", type=Path, default=Path("results/submission_jrtip"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.submission_dir.resolve()
    output = (args.output or root / "submission_audit.json").resolve()

    tex_path = root / "manuscript.tex"
    bib_path = root / "references.bib"
    pdf_path = root / "manuscript.pdf"
    archive_path = root / "jrtip_submission_sources.zip"
    for path in (tex_path, bib_path, pdf_path, archive_path):
        if not path.is_file():
            raise RuntimeError(f"required generated artifact missing: {path}")

    tex = tex_path.read_text(encoding="utf-8")
    bib = bib_path.read_text(encoding="utf-8")
    abstract = re.sub(r"\\[A-Za-z]+(?:\{[^}]*\})?", " ", extract(tex, "\\begin{abstract}", "\\end{abstract}"))
    abstract_words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", abstract)
    keywords_raw = extract(tex, "\\keywords{", "}")
    keywords = [item.strip() for item in keywords_raw.split("\\and") if item.strip()]

    citations: set[str] = set()
    for group in re.findall(r"\\cite(?:t|p)?\{([^}]+)\}", tex):
        citations.update(key.strip() for key in group.split(","))
    bib_keys = set(re.findall(r"^@[A-Za-z]+\{([^,]+),", bib, re.M))

    pdfinfo = subprocess.run(["pdfinfo", str(pdf_path)], check=True, capture_output=True, text=True).stdout
    page_match = re.search(r"^Pages:\s+(\d+)", pdfinfo, re.M)
    if not page_match:
        raise RuntimeError("pdfinfo did not report a page count")
    pages = int(page_match.group(1))
    pdffonts = subprocess.run(["pdffonts", str(pdf_path)], check=True, capture_output=True, text=True).stdout
    font_rows = [
        match.groups() for line in pdffonts.splitlines()
        if (match := re.search(r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+$", line))
    ]
    fonts_embedded = bool(font_rows) and all(row[0] == "yes" for row in font_rows)

    with zipfile.ZipFile(archive_path) as archive:
        archive_names = set(archive.namelist())
    flat_archive = all("/" not in name.rstrip("/") for name in archive_names)
    missing_files = sorted(REQUIRED_FILES - archive_names)

    required_sections = [
        "Data availability", "Funding", "Competing interests",
        "Author contributions", "Use of generative AI",
    ]
    missing_sections = [name for name in required_sections if name not in tex]
    placeholders = sorted(set(re.findall(r"[A-Z][A-Z ()/-]{3,}(?:REQUIRED|CONFIRMATION REQUIRED)", tex)))
    blank_fields = re.findall(r"\\blankfield\{([^}]+)\}", tex)

    checks = {
        "abstract_150_to_250_words": 150 <= len(abstract_words) <= 250,
        "keywords_4_to_6": 4 <= len(keywords) <= 6,
        "page_count_at_most_12": pages <= 12,
        "all_pdf_fonts_embedded": fonts_embedded,
        "all_citations_have_bibliography_entries": not (citations - bib_keys),
        "all_bibliography_entries_are_cited": not (bib_keys - citations),
        "mandatory_declaration_sections_present": not missing_sections,
        "source_archive_is_flat": flat_archive,
        "source_archive_complete": not missing_files,
    }
    report = {
        "status": "PASS_WITH_AUTHOR_INPUT_PENDING" if all(checks.values()) and (placeholders or blank_fields) else "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metrics": {
            "abstract_words": len(abstract_words),
            "keyword_count": len(keywords),
            "page_count": pages,
            "citation_count": len(citations),
            "bibliography_entry_count": len(bib_keys),
        },
        "missing_bibliography_keys": sorted(citations - bib_keys),
        "uncited_bibliography_keys": sorted(bib_keys - citations),
        "missing_declaration_sections": missing_sections,
        "missing_source_files": missing_files,
        "author_input_placeholders": placeholders,
        "blank_author_fields": blank_fields,
        "artifacts": {
            "manuscript_pdf": {"path": pdf_path.name, "sha256": sha256(pdf_path)},
            "source_archive": {"path": archive_path.name, "sha256": sha256(archive_path)},
        },
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    bundle_path = root / "jrtip_pre_submission_bundle.zip"
    bundle_files = [
        pdf_path, archive_path, root / "cover_letter.md", root / "submission_checklist.md",
        root / "submission_manifest.json", output,
    ]
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in bundle_files:
            archive.write(path, arcname=path.name)
    print(json.dumps(report, indent=2))
    print(f"pre_submission_bundle={bundle_path}")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
