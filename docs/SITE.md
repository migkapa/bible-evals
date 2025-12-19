# Landing Page (static)

The repository includes a static landing page in `site/` that visualizes saved evaluation runs.

## Where results are saved

Every `bible-eval run` writes:

- ephemeral run artifacts: `runs/<run_id>/...` (usually gitignored)
- durable history: `results/history.json`
- durable per-run copy: `results/runs/<run_id>.json`
- durable per-verse details: `results/details/<run_id>/<model_slug>.json`
- site data mirror: `site/data/history.json`
  - plus example data: `site/data/details/<run_id>/<model_slug>.json`

## Build / update site data

If you have a history already:

```bash
bible-eval export-site --history results/history.json --out site
```

## View locally

Browsers typically block `fetch()` from `file://` pages, so serve it:

```bash
cd site
python -m http.server 8000
```

Then open `http://localhost:8000`.

## GitHub Pages

You can publish the `site/` folder with GitHub Pages (or a workflow) since it is fully static.
