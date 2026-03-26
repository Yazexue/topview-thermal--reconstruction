# Repository Layout

```text
.
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── requirements.txt
├── pyproject.toml
├── docs/
│   ├── REPRODUCTION_GUIDE.md
│   ├── REPOSITORY_LAYOUT.md
│   └── SUBMISSION_CHECKLIST.md
├── notebooks/
│   └── TOPVIEW_Colab_Workflow.ipynb
└── src/
    └── topview/
        ├── config.py
        ├── io.py
        ├── manifest.py
        ├── legacy_manifest.py
        ├── splits.py
        ├── clouds.py
        ├── datasets.py
        ├── baselines.py
        ├── evaluation.py
        ├── reporting.py
        ├── training.py
        ├── utils.py
        ├── models/
        └── cli/
```

## Top-Level Purpose

- `README.md`: project overview and primary commands.
- `CHANGELOG.md`: versioned repository changes.
- `CONTRIBUTING.md`: collaboration and update conventions.
- `docs/`: reproduction and handoff material.
- `notebooks/`: Colab execution template.
- `src/topview/`: implementation package.

## Data and Outputs

Runtime data and generated artifacts are intentionally kept outside version control.
They are written under the configured runtime root into folders such as:

- `manifests/`
- `splits/`
- `runs/`
- `reports/`
- `exports/`
