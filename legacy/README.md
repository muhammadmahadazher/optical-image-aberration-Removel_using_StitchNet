# Original prototype

This folder preserves the files and weights from the repository before the v2 rebuild. They remain available for audit and comparison but are not imported, tested, or executed by the maintained application.

The original inference path compared each tile with itself, used overlap slices based on step size rather than overlap width, ignored estimated translations during composition, and produced large holes in the included example report. The two supplied 49.8 MB weight files were byte-identical and produced a destructive non-zero warp even for identical image pairs.

Use the root `python start.py` launcher or the installed `stitchnet` CLI. Do not use these legacy scripts or weights for research output.
