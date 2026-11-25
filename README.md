# Personalising language embeddings reveals structured individual differences in how humans represent meaning
**Fabian Schneider, & Helen Blank**

### Abstract
*People differ in the meanings they associate with words, but how these differences are organised across concepts remains unclear. Here we tested whether individual semantic judgements reveal stable, generalisable structure and whether personalising language embeddings can capture it. We measured semantic geometries using multiple-arrangement judgements across multiple sessions. Participants’ judgements were more similar within and across sessions than to those of other people or embeddings. Adapting embeddings to individuals using ten minutes of behavioural data improved prediction of held-out judgements by approximately 40\%, with benefits extending to word pairs and words absent from training. Demographic and experiential characteristics predicted both embedding-human divergence and individual departures from shared semantic structure. Individual departures varied systematically across semantic domains and word properties. These findings show that individual semantic differences contain stable, learnable structure that extends across concepts and is associated with demographic, experiential and word characteristics.*

Read the full paper [here](#blank).

## Data and experiments

| Dataset | Design | Analysis directory |
|---|---|---|
| Experiment 1 | 30 participants, two sessions. Within- and between-session reliability, adaptation and generalisation. | `analysis/data/ds-exp1/` |
| Experiment 2 | 303 participants, one session and a domain-interest questionnaire. Replication, demographic analyses and comparison of linear and non-linear adapters. | `analysis/data/ds-exp2/` |

Both experiments used 70 words presented on boards of 20. The [experiment code](experiments/) contains the browser tasks. The [analysis code](analysis/) reads the prepared datasets and writes derivatives under each experiment directory.

**Data release**: [DOI](#blank)

Place the prepared experiment directories under `analysis/data/` and the shared resources under `analysis/data/resources/`. When extracting the derivatives, preserve their directory structure. To regenerate the figures from existing fits, download the saved task outputs. 

## Installation
From a terminal with Git and Python installed:

```bash
git clone https://github.com/FabulousFabs/personal-embeddings.git
cd personal-embeddings
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The analyses use PyTorch, Hydra and [MVPy](https://github.com/FabulousFabs/MVPy). To run the figure notebooks locally, install JupyterLab in the same environment:

```bash
python -m pip install jupyterlab
```

## Reproducing an analysis

Run [main.py](analysis/main.py) **from the `analysis/` directory** so the relative data paths resolve correctly. Hydra selects the experiment (`data`), adapter architecture (`model`) and tasks. Each task has a `skip` flag. Training is enabled by default, so disable it when running a standalone behavioural analysis.

For example, to compute within- and between-session reliability for Experiment 1 using GloVe, start from the repository root:

```bash
cd analysis
python main.py data=exp1 run=101 device=cpu \
    'data.embeddings.models=[glove]' \
    task.mareliability.skip=false \
    task.macrossreliability.skip=false \
    task.matrain.skip=true
```

This writes two files under `data/ds-exp1/derivatives/run-101/tasks/`:

```text
task-mareliability_bse0-glove_mod-delta.pt
task-macrossreliability_bse0-glove_mod-delta.pt
```

The filename includes the configured model name even for behavioural reliability. Run `101` keeps the example separate from the archived runs used by the notebooks.

To repeat the analysis for another embedding, replace `glove` with `t5-large`, `t5-large-context` or `e5-large`. For Experiment 2, select `data=exp2`, enable `task.mareliability.skip=false` and keep `task.matrain.skip=true`. Leave cross-session reliability disabled because these participants completed one session.

## Reproducing a figure

For example, Figure 6 shows which word properties predict the advantage of non-linear over linear personalisation for GloVe. The notebook reads the following task output:

```text
analysis/data/ds-exp2/derivatives/run-006/tasks/task-maspace_bse0-glove_mod-delta.pt
```

From the repository root:

```bash
cd analysis/figures
mkdir -p deliverable/png deliverable/pdf deliverable/svg
python -m jupyterlab fig6.ipynb
```

Run the cells in order. The notebook writes PNG, PDF and SVG versions under `deliverable/`. Launch the other figure notebooks from the same `analysis/figures/` directory.

## Citation
If you use the code or datasets provided here, please cite our paper:
```text
Schneider, F., & Blank, H. (2026). Personalising language embeddings reveals structured individual differences in how humans represent meaning.
```

## License
This project is licensed under [license](#blank).