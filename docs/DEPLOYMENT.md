# Hosted preview deployment

The GitHub Pages site is a read-only, static demonstration built from `frontend/`. It contains one precomputed, public OpenSlide H&E test fixture and its machine-readable quality report. It does not accept uploads or run the Python, OpenCV, or RAFT backend in GitHub Pages.

Every deployment installs the locked frontend dependencies, runs its production test/build, uploads the resulting artifact, and then deploys through the protected `github-pages` environment. The workflow runs on `main` changes to the frontend and can also be dispatched manually.

## Rollback

Open the repository's **Actions → Pages demo**, select the last known-good commit, and rerun the workflow for that revision. If the preview must be removed immediately, use **Settings → Pages → Unpublish site**. The local application and model artifacts are independent of Pages.
