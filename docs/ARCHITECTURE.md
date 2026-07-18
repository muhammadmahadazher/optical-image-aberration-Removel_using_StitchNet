# Architecture

## System flow

```mermaid
flowchart LR
  UI["React/Vite local UI"] --> API["FastAPI validation layer"]
  API --> JOB["Persistent single-worker job queue"]
  JOB --> IO["Tile discovery and normalization"]
  IO --> REG["Pairwise hybrid or learned-gated registration"]
  REG --> OPT["Robust global layout optimizer"]
  OPT --> BLEND["Exposure compensation and feather blend"]
  BLEND --> GATE["Coverage and quality gate"]
  GATE --> OUT["PNG, preview, coverage mask, JSON report"]
```

## Core invariants

1. **The grid is explicit.** Coordinates come from row/column filenames or a validated indexed layout. Duplicate or mixed layouts fail before image processing.
2. **Registration is evidence, not authority.** Each neighbor constraint carries its method, confidence, inlier count, overlap error, and optimizer usage flag.
3. **Global consistency wins.** Soft-L1 graph optimization combines pair evidence with low-weight nominal stage priors and rejects inconsistent constraints.
4. **Black tissue is valid data.** Cropping uses only the coverage mask, never pixel intensity.
5. **Memory is bounded before allocation.** Estimated canvas megapixels are checked against configuration.
6. **Learned output cannot silently regress.** RAFT corrections are accepted per image only if photometric overlap improves; identity inputs and rejected corrections return the original tensor exactly.
7. **The default remains deterministic.** Hybrid SIFT/correlation registration does not require the GPU or learned checkpoint.

## Backend

`backend.app` exposes versioned `/api/v1` endpoints. Multipart inputs stream to a UUID-scoped job directory. ZIP extraction rejects traversal, absolute paths, drive prefixes, symbolic links, encryption, excessive entry counts, excessive uncompressed size, and suspicious compression ratios.

`JobManager` persists a manifest after every state transition. Jobs found in an in-flight state after restart are marked failed with an interruption reason rather than silently resumed. One worker is the safe default to cap memory; `STITCHNET_JOB_WORKERS` is available for controlled research systems.

Artifacts:

- `mosaic.png`: 8-bit or 16-bit full result.
- `preview.jpg`: bounded browser preview.
- `coverage.png`: provenance mask for valid output pixels.
- `quality-report.json`: configuration, metrics, constraints, warnings, runtime, and intended-use notice.

## Registration

The hybrid path builds normalized cross-correlation and SIFT candidates inside plausible overlap bands. Feature matches are filtered by descriptor ratio, nominal displacement window, and robust median consensus. Candidates are scored by confidence and robust photometric disagreement; agreeing methods are confidence-weighted. Low-confidence evidence falls back to nominal placement and remains visible in the report.

The optional learned path samples square windows along overlap bands, runs the passed RAFT-small checkpoint, rejects non-improving windows, aggregates accepted median flow, and adds that candidate to the same classical arbitration. Loading is refused if checkpoint metadata says its deployment gate failed.

## Frontend

The frontend is a small React/Vite client with no cloud service dependency. It polls health and job status, never embeds credentials, and sends images only to the configurable local API URL. The one-command launcher stages dependencies in a user cache to avoid sync-folder filesystem failures.

## Portability

The maintained Python package supports Python 3.10–3.12. The launcher resolves `npm.cmd` on Windows and `npm` on POSIX, propagates API configuration through `VITE_API_URL`, handles Ctrl+C, checks port conflicts, and terminates child processes in reverse order.

## Deliberate limits

- The engine reconstructs 2D regular or partially missing grids; it is not a free-form panorama or 3D stack stitcher.
- It does not perform cell segmentation, tumor classification, counting, or diagnosis.
- Learned evaluation uses controlled synthetic aberrations on real microscopy content; this is not a substitute for instrument-specific paired calibration data.
- Clinical use requires locked acquisition protocols, representative multi-site validation, human-factors testing, traceability, change control, and applicable regulatory review.
