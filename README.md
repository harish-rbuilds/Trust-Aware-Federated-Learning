# Trust-Aware Differential Privacy for Federated Learning

A five-client federated-learning experiment on the scikit-learn Digits dataset.
The experiment compares fixed Gaussian update noise (`sigma = 0.011`) with
trust-aware Gaussian noise assigned from predefined client trust scores.

## Repository layout

```text
Trust-Aware-Federated-Learning/
├── code/
│   └── zkdp_fl_experiment.py
├── visualization/
│   └── visualize_results.py
├── data/
│   └── README.md
├── results/
│   └── README.md
├── figures/
│   └── README.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Run

```bash
pip install -r requirements.txt
python code/zkdp_fl_experiment.py
python visualization/visualize_results.py
```

The experiment writes CSV files to `results/`. The visualization script reads
those CSV files and writes PNG plots to `figures/`.

## Scope notes

The current script implements clipping, Gaussian update noise, trust-to-noise
mapping, FedAvg aggregation, and evaluation. Trust scores are predefined
experimental inputs. The noise parameters are not calibrated to a formal
`(epsilon, delta)` privacy budget. Zero-knowledge proof verification and
compute-aware aggregation are not implemented in this prototype.
