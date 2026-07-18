"use client";

import {
  ChangeEvent,
  CSSProperties,
  DragEvent,
  FormEvent,
  KeyboardEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

const API_BASE =
  (import.meta.env.VITE_API_URL as string | undefined) ?? "http://127.0.0.1:8000";
const STATIC_DEMO = import.meta.env.VITE_STATIC_DEMO === "true";
const DEMO_BASE = import.meta.env.BASE_URL + "demo/";
const ACCEPTED = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".zip"];

type QualityReport = {
  status: "pass" | "review" | "failed";
  tile_count: number;
  grid_rows: number;
  grid_columns: number;
  output_width: number;
  output_height: number;
  registration_count: number;
  fallback_count: number;
  median_confidence: number;
  placement_rmse_px: number;
  placement_p95_px: number;
  seam_mae: number;
  coverage_fraction: number;
  runtime_seconds?: number;
  warnings: string[];
};

type Job = {
  id: string;
  status: "uploading" | "queued" | "running" | "succeeded" | "failed";
  progress: number;
  message: string;
  report: QualityReport | null;
  error: string | null;
  artifacts: {
    preview: string;
    result: string;
    coverage: string;
    report: string;
  } | null;
};

type Health = {
  status: string;
  cuda: boolean;
  device: string | null;
  learned_model: {
    available: boolean;
    quality_gate_passed: boolean;
    checkpoint: string;
  };
};

function formatBytes(bytes: number) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function percent(value: number, digits = 0) {
  return (value * 100).toFixed(digits) + "%";
}

function apiUrl(path: string) {
  return path.startsWith("http") || STATIC_DEMO ? path : API_BASE + path;
}

const DEMO_REPORT: QualityReport = {
  status: "pass",
  tile_count: 9,
  grid_rows: 3,
  grid_columns: 3,
  output_width: 955,
  output_height: 955,
  registration_count: 12,
  fallback_count: 0,
  median_confidence: 1,
  placement_rmse_px: 0.1081261954,
  placement_p95_px: 0.1586489612,
  seam_mae: 0.0115393441,
  coverage_fraction: 1,
  runtime_seconds: 12.196,
  warnings: [],
};

const DEMO_JOB: Job = {
  id: "00000000000000000000000000000001",
  status: "succeeded",
  progress: 1,
  message: "Verified H&E sample loaded",
  report: DEMO_REPORT,
  error: null,
  artifacts: {
    preview: DEMO_BASE + "preview.jpg",
    result: DEMO_BASE + "mosaic.png",
    coverage: DEMO_BASE + "coverage.png",
    report: DEMO_BASE + "quality-report.json",
  },
};

export default function Home() {
  const fileInput = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [overlap, setOverlap] = useState(STATIC_DEMO ? 25 : 10);
  const [columns, setColumns] = useState(STATIC_DEMO ? "3" : "");
  const [registration, setRegistration] = useState(STATIC_DEMO ? "learned" : "hybrid");
  const [bitDepth, setBitDepth] = useState<8 | 16>(8);
  const [dragging, setDragging] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState("");

  const totalBytes = useMemo(() => files.reduce((sum, file) => sum + file.size, 0), [files]);
  const busy = job?.status === "uploading" || job?.status === "queued" || job?.status === "running";
  const report = job?.report;

  useEffect(() => {
    if (STATIC_DEMO) {
      setHealth({
        status: "demo",
        cuda: false,
        device: "Curated H&E sample",
        learned_model: {
          available: true,
          quality_gate_passed: true,
          checkpoint: "raft_microscopy_v2.pt",
        },
      });
      return;
    }
    let mounted = true;
    const check = () => {
      fetch(API_BASE + "/api/v1/health")
        .then((response) => {
          if (!response.ok) throw new Error("Backend unavailable");
          return response.json();
        })
        .then((data: Health) => mounted && setHealth(data))
        .catch(() => mounted && setHealth(null));
    };
    check();
    const timer = window.setInterval(check, 15000);
    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (STATIC_DEMO) return;
    const restoredJob = new URLSearchParams(window.location.search).get("job");
    if (!restoredJob || !/^[a-f0-9]{32}$/.test(restoredJob)) return;
    fetch(API_BASE + "/api/v1/jobs/" + restoredJob)
      .then((response) => {
        if (!response.ok) throw new Error("Saved job is no longer available.");
        return response.json();
      })
      .then((saved: Job) => setJob(saved))
      .catch((restoreError) =>
        setError(restoreError instanceof Error ? restoreError.message : "Could not restore job."),
      );
  }, []);

  useEffect(() => {
    if (!job || !["uploading", "queued", "running"].includes(job.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const response = await fetch(API_BASE + "/api/v1/jobs/" + job.id);
        if (!response.ok) throw new Error("Could not read job status");
        const next = (await response.json()) as Job;
        setJob(next);
        if (next.status === "failed") setError(next.error || "Reconstruction failed.");
      } catch (pollError) {
        setError(pollError instanceof Error ? pollError.message : "Connection interrupted.");
      }
    }, 700);
    return () => window.clearInterval(timer);
  }, [job?.id, job?.status]);

  function addFiles(incoming: File[]) {
    setError("");
    const accepted = incoming.filter((file) => {
      const lower = file.name.toLowerCase();
      return ACCEPTED.some((extension) => lower.endsWith(extension));
    });
    if (!accepted.length) {
      setError("Choose PNG, JPEG, TIFF, BMP, or ZIP microscopy tiles.");
      return;
    }
    const unique = new Map(files.map((file) => [file.name + ":" + file.size, file]));
    accepted.forEach((file) => unique.set(file.name + ":" + file.size, file));
    setFiles(Array.from(unique.values()));
    setJob(null);
  }

  function onFileInput(event: ChangeEvent<HTMLInputElement>) {
    addFiles(Array.from(event.target.files ?? []));
    event.target.value = "";
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    addFiles(Array.from(event.dataTransfer.files));
  }

  function onDropzoneKey(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.current?.click();
    }
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (STATIC_DEMO) {
      setError("");
      setJob(DEMO_JOB);
      return;
    }
    if (!files.length) {
      setError("Add at least one tile or ZIP archive first.");
      return;
    }
    setError("");
    setJob({
      id: "pending",
      status: "uploading",
      progress: 0,
      message: "Uploading tiles securely",
      report: null,
      error: null,
      artifacts: null,
    });
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    form.append(
      "configuration",
      JSON.stringify({
        overlap: overlap / 100,
        columns: columns ? Number(columns) : null,
        registration,
        output_bit_depth: bitDepth,
        compensate_exposure: true,
        crop_to_valid_region: true,
        allow_missing_tiles: true,
      }),
    );
    try {
      const response = await fetch(API_BASE + "/api/v1/jobs", { method: "POST", body: form });
      const payload = await response.json();
      if (!response.ok) {
        const detail = payload.detail?.message ?? payload.detail ?? "Upload validation failed.";
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      const created = payload as Job;
      setJob(created);
      window.history.replaceState(null, "", "?job=" + created.id);
    } catch (submitError) {
      const message = submitError instanceof Error ? submitError.message : "Could not start reconstruction.";
      setError(message);
      setJob(null);
    }
  }

  const qualityTone = report?.status === "pass" ? "good" : report?.status === "review" ? "review" : "neutral";

  return (
    <main className="app-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />

      <header className="topbar glass">
        <a className="brand" href="#" aria-label="StitchNet Laboratory home">
          <span className="brand-mark"><span /></span>
          <span>
            <strong>StitchNet</strong>
            <small>Laboratory</small>
          </span>
        </a>
        <div className="topbar-center">
          <span className="research-badge"><i /> Research use only</span>
          <span className="privacy-note">{STATIC_DEMO ? "Read-only hosted preview" : "Images stay on this machine"}</span>
        </div>
        <div className={"system-state " + (health ? "online" : "offline")}>
          <span className="state-dot" />
          <span>{STATIC_DEMO ? "Demo ready" : health ? "Engine ready" : "Engine offline"}</span>
          <small>{STATIC_DEMO ? "Verified public sample" : health?.cuda ? health.device ?? "CUDA" : health ? "CPU mode" : "Start the local service"}</small>
        </div>
      </header>

      <section className="intro">
        <div>
          <p className="eyebrow">Microscopy reconstruction workspace</p>
          <h1>Turn overlapping fields into one trustworthy view.</h1>
          <p className="lede">
            Confidence-aware registration, global stage correction, and seam-balanced blending
            for analysis-ready cell mosaics.
          </p>
        </div>
        <div className="intro-facts" aria-label="Pipeline features">
          <span><b>01</b> Pairwise evidence</span>
          <span><b>02</b> Global consistency</span>
          <span><b>03</b> Quality report</span>
        </div>
      </section>

      <form className="workspace" onSubmit={submit}>
        <aside className="control-panel glass">
          <div className="panel-heading">
            <span className="step-number">01</span>
            <div><p>Source</p><h2>Add image tiles</h2></div>
          </div>

          <input
            ref={fileInput}
            className="visually-hidden"
            type="file"
            accept={ACCEPTED.join(",")}
            multiple
            disabled={STATIC_DEMO}
            onChange={onFileInput}
          />
          <div
            className={"dropzone " + (dragging ? "is-dragging" : "")}
            role="button"
            tabIndex={STATIC_DEMO ? -1 : 0}
            aria-disabled={STATIC_DEMO}
            onKeyDown={onDropzoneKey}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={STATIC_DEMO ? undefined : onDrop}
            onClick={() => !STATIC_DEMO && fileInput.current?.click()}
            aria-label={STATIC_DEMO ? "Hosted demonstration sample" : "Choose or drop microscopy tiles"}
          >
            <span className="upload-glyph">{STATIC_DEMO ? "◇" : "+"}</span>
            <strong>{STATIC_DEMO ? "Verified H&E sample" : files.length ? files.length + " source file" + (files.length > 1 ? "s" : "") : "Drop tiles or a ZIP"}</strong>
            <small>{STATIC_DEMO ? "3 × 3 · learned + gated" : files.length ? formatBytes(totalBytes) + " selected" : "PNG · TIFF · JPEG · BMP"}</small>
          </div>

          {files.length > 0 && (
            <div className="file-summary">
              <div className="file-stack" aria-hidden="true">
                {files.slice(0, 4).map((file, index) => <span key={file.name + index}>{index + 1}</span>)}
              </div>
              <button type="button" onClick={() => { setFiles([]); setJob(null); }}>Clear selection</button>
            </div>
          )}

          <div className="divider" />
          <div className="panel-heading compact">
            <span className="step-number">02</span>
            <div><p>Method</p><h2>Reconstruction</h2></div>
          </div>

          <label className="field-label" htmlFor="overlap">
            <span>Expected overlap</span><output>{overlap}%</output>
          </label>
          <input
            id="overlap"
            className="range"
            type="range"
            min="2"
            max="50"
            value={overlap}
            disabled={STATIC_DEMO}
            onChange={(event) => setOverlap(Number(event.target.value))}
          />
          <div className="range-scale"><span>2%</span><span>50%</span></div>

          <div className="field-grid">
            <label>
              <span>Grid columns <em>optional</em></span>
              <input
                type="number"
                min="1"
                placeholder="Auto"
                value={columns}
                disabled={STATIC_DEMO}
                onChange={(event) => setColumns(event.target.value)}
              />
            </label>
            <label>
              <span>Registration</span>
              <select disabled={STATIC_DEMO} value={registration} onChange={(event) => setRegistration(event.target.value)}>
                <option value="hybrid">Hybrid</option>
                <option
                  value="learned"
                  disabled={!health?.learned_model.quality_gate_passed}
                >
                  Learned + gated
                </option>
                <option value="features">Features</option>
                <option value="correlation">Correlation</option>
                <option value="nominal">Grid only</option>
              </select>
            </label>
          </div>

          <div className="bit-depth">
            <span>Output depth</span>
            <div role="group" aria-label="Output bit depth">
              <button disabled={STATIC_DEMO} type="button" className={bitDepth === 8 ? "active" : ""} onClick={() => setBitDepth(8)}>8-bit</button>
              <button disabled={STATIC_DEMO} type="button" className={bitDepth === 16 ? "active" : ""} onClick={() => setBitDepth(16)}>16-bit</button>
            </div>
          </div>

          <button className="primary-action" type="submit" disabled={STATIC_DEMO ? false : busy || !files.length || !health}>
            <span>{STATIC_DEMO ? job?.status === "succeeded" ? "Reload verified sample" : "Explore live sample" : busy ? "Reconstructing" : "Build mosaic"}</span>
            <i>{busy ? Math.round((job?.progress ?? 0) * 100) + "%" : "→"}</i>
          </button>
          {STATIC_DEMO && <p className="hosted-note">This Pages preview uses a precomputed, source-linked public test fixture. Clone the project to process your own tiles locally.</p>}
          {error && <p className="error-message" role="alert">{error}</p>}
          <p className="use-note">
            For research preprocessing only. Review every quality flag before downstream analysis.
          </p>
        </aside>

        <section className="viewer glass">
          <div className="viewer-bar">
            <div>
              <span className="viewer-kicker">Mosaic canvas</span>
              <strong>{job?.status === "succeeded" ? "Reconstruction result" : files.length ? "Ready to reconstruct" : "No slide loaded"}</strong>
            </div>
            <div className={"status-pill " + qualityTone}>
              <i />
              {report ? report.status === "pass" ? "Quality checks passed" : "Review suggested" : job ? job.message : "Awaiting tiles"}
            </div>
          </div>

          <div className={"canvas " + (job?.status === "succeeded" ? "has-result" : "")}>
            {job?.status === "succeeded" && job.artifacts ? (
              <img src={apiUrl(job.artifacts.preview) + "?v=" + job.id} alt="Stitched microscopy mosaic preview" />
            ) : (
              <div className="empty-canvas">
                <div className={"tile-orbit " + (busy ? "processing" : "")} aria-hidden="true">
                  <span /><span /><span /><span /><span /><span /><span /><span /><span />
                  <b />
                </div>
                <strong>{busy ? job?.message : files.length ? "Configuration ready" : "Your mosaic will appear here"}</strong>
                <p>{busy ? "The quality gate runs before the result is released." : "Add overlapping tiles with row/column or indexed filenames."}</p>
              </div>
            )}
            {busy && (
              <div className="progress-track" aria-label={"Progress " + Math.round((job?.progress ?? 0) * 100) + "%"}>
                <span style={{ width: percent(job?.progress ?? 0) }} />
              </div>
            )}
          </div>

          <div className="viewer-footer">
            <div className="view-tools">
              <button type="button" className="active">Fit</button>
              <span>Pixel-preserving preview</span>
            </div>
            {job?.status === "succeeded" && job.artifacts ? (
              <a className="download-action" href={apiUrl(job.artifacts.result)} download>Download full mosaic <span>↓</span></a>
            ) : (
              <span className="canvas-hint">Results include a PNG, mask, and JSON report</span>
            )}
          </div>
        </section>

        <aside className="quality-panel glass">
          <div className="panel-heading">
            <span className="step-number">03</span>
            <div><p>Evidence</p><h2>Quality gate</h2></div>
          </div>

          <div className={"quality-score " + qualityTone}>
            <div className="score-ring" style={{ "--score": report ? Math.round(report.median_confidence * 100) : 0 } as CSSProperties}>
              <span>{report ? Math.round(report.median_confidence * 100) : "—"}</span>
              <small>{report ? "%" : ""}</small>
            </div>
            <div>
              <strong>{report ? "Registration confidence" : "Not measured yet"}</strong>
              <p>{report ? report.registration_count + " neighboring relationships evaluated" : "Run a reconstruction to populate evidence."}</p>
            </div>
          </div>

          <div className="metric-list">
            <div><span>Placement p95</span><strong>{report ? report.placement_p95_px.toFixed(2) + " px" : "—"}</strong></div>
            <div><span>Seam difference</span><strong>{report ? report.seam_mae.toFixed(4) : "—"}</strong></div>
            <div><span>Covered output</span><strong>{report ? percent(report.coverage_fraction, 1) : "—"}</strong></div>
            <div><span>Fallback pairs</span><strong>{report ? report.fallback_count : "—"}</strong></div>
          </div>

          <div className="quality-divider" />
          <div className="dataset-card">
            <span>Output geometry</span>
            <strong>{report ? report.output_width.toLocaleString() + " × " + report.output_height.toLocaleString() : "Pending"}</strong>
            <div>
              <small>{report ? report.grid_rows + " × " + report.grid_columns + " grid" : "Grid —"}</small>
              <small>{report ? report.tile_count + " tiles" : "Tiles —"}</small>
              <small>{report?.runtime_seconds ? report.runtime_seconds.toFixed(1) + " s" : "Runtime —"}</small>
            </div>
          </div>

          <div className="review-list">
            <span>Review notes</span>
            {report?.warnings.length ? report.warnings.slice(0, 3).map((warning) => (
              <p key={warning}><i className="warning-dot" />{warning}</p>
            )) : (
              <p><i className={report ? "check-dot" : "empty-dot"} />{report ? "No blocking quality flags." : "No report available."}</p>
            )}
          </div>

          {job?.status === "succeeded" && job.artifacts && (
            <a className="report-link" href={apiUrl(job.artifacts.report)} target="_blank" rel="noreferrer">
              Open complete quality report <span>↗</span>
            </a>
          )}
        </aside>
      </form>

      <footer className="footer">
        <span>StitchNet Laboratory · {STATIC_DEMO ? "hosted read-only preview" : "local research workspace"}</span>
        <span>Confidence is evidence, not a diagnosis.</span>
      </footer>
    </main>
  );
}
