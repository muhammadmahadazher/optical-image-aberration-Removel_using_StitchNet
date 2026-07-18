# Medical safety and intended use

## Intended use

StitchNet Laboratory is research software for reconstructing overlapping two-dimensional microscopy tiles into a reviewable mosaic. It can be used as a preprocessing step in exploratory cell-imaging workflows.

## Not intended for

- cancer detection, grading, staging, prognosis, or treatment decisions;
- autonomous cell segmentation or counting;
- primary display or archival of patient records;
- replacing a pathologist, laboratory scientist, or validated microscope workflow;
- use as a clinically validated medical device.

The source H&E slide used for open-world testing demonstrates content transfer only. It does not contain task annotations and cannot establish diagnostic sensitivity, specificity, or clinical utility.

## Risk controls implemented

- Local-only image transfer in the supplied launcher and UI.
- Explicit research-only notices in API metadata, UI, reports, and documentation.
- No silent learned correction: per-input no-regression gate plus failed-checkpoint refusal.
- Constraint-level provenance and output coverage mask.
- Input, archive, canvas-size, and configuration validation.
- Deterministic seeds and source-disjoint evaluation splits.
- Exact identity behavior for learned correction when evidence is absent.
- No deployment workflow and no automatic external uploads.

## Required review before downstream analysis

1. Confirm tile order, grid dimensions, overlap, channel, bit depth, and acquisition series.
2. Inspect the full-resolution mosaic, not only the JPEG preview.
3. Review coverage, placement p95, seam difference, confidence, fallbacks, rejected constraints, and warnings.
4. Compare landmarks across several seams, including low-texture regions and borders.
5. Retain the JSON report, coverage mask, source hashes or identifiers, software commit, and configuration.
6. Reject or reacquire data when quality evidence is weak.

## Path toward clinical validation

A separate program would be required: define the clinical claim, curate representative annotated cohorts across sites, scanners, and stains; prevent patient/source leakage; pre-register acceptance thresholds; compare against an appropriate reference standard; quantify uncertainty and failure rates; conduct prospective and human-factors studies; implement data governance and cybersecurity controls; and follow the relevant regulatory and quality-management process.
