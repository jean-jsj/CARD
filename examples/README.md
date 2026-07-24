# examples/

Hands-on entry points, in order. The notebooks run on the ~18 MB `dev_mini`
starter slices, so each executes in minutes on a laptop or free Colab.

| | Notebook | What it covers |
|---|---|---|
| 1 | [01_explore_data.ipynb](01_explore_data.ipynb) | every released file: the panel, prices/promotions, the product text and the structure it carries, the instruments, the 16 scored scenarios |
| 2 | [02_score_a_submission.ipynb](02_score_a_submission.ipynb) | the full loop: build the three submission CSVs from a naive model, score against hidden truth, read the report |
| 3 | [03_endogeneity_and_instruments.ipynb](03_endogeneity_and_instruments.ipynb) | the core phenomenon live: naive vs instrumented estimates across the paired endogeneity on/off cells |

Each notebook carries an *Open in Colab* badge; on Colab it clones the repo and
fetches the mini data by itself.

Scripts:

- [`download_data.py`](download_data.py) — fetch a cell from Hugging Face
  (`--mini` for the starter slice).
- [`quickstart.py`](quickstart.py) — the naive baseline + scoring of notebook 2
  as one command.
