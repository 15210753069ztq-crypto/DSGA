# DSGA: Drift-Aware Selective Graph Adaptation

Code and datasets for *Not Every Drift Changes the Graph: Drift-Aware Selective Adaptation for
Graph-Based Continual Multivariate Time-Series Forecasting*.

DSGA diagnoses, after every revealed block, whether the data distribution (marginal evidence) or the
cross-variable predictability (dependency evidence) has changed relative to the warm-up data, and routes
the block to one of four actions: `freeze`, `time`, `graph`, `joint`. DSGA also uses low-rank weight
modulation and a second, instance-normalized view of the same backbone, combined either by per-block
switching (S) or by an exponentiated-gradient ensemble (E).

## Layout

```
experiments/
  run_pems08_prequential_multihorizon.py   DSGA: model, diagnosis, routing, memory, dual view, online loop
  run_natural_stream_benchmark.py          null calibration and drift diagnosis (marginal / dependency scores)
  run_neural_gnn_prediction.py             scores, seeds, shared utilities
  run_neural_gnn_prediction_v2.py          per-action trainable subsets and learning rates
  e0_oracle/run_e0.py                      ridge helpers used by the dependency score
  run_pa_protocol.py                       protocol, hyperparameter grid, GPU scheduler (search / select / final)
  run_pa_default.py                        DSGA with the default configuration
  run_pa_dvablation.py                     dual-view variants: mod, ens (frozen ensemble), ens_full (= E)
  run_pa_modesel.py                        validation-stream choice between S and E
  run_pa_sca_ablation.py                   routing ablation on the controlled stream (5 streams x variants)
dataset/                                   the nine streams and their preparation scripts (see dataset/README.md)
```

## Setup

```
pip install -r requirements.txt
python dataset/prepare_data.py          # unpack the large .csv.gz streams
# Air-Stream is not shipped: python dataset/prepare_eac_streams.py download && python dataset/prepare_eac_streams.py build air
```

Python 3.12 and CUDA 12.4 were used; the runner also works on CPU (`--device cpu`).

## Protocol

Each stream is split chronologically 20% / 5% / 75% (train / validation / online). The lookback is
`L=60` and the horizons are `H=1` and `H=24`. The online part is divided into eight blocks. A block is
predicted with fixed parameters, then its labels are released; only windows whose full `H`-step target
has been released are used for diagnosis-triggered updates and for view selection.

## Reproducing the DSGA results

All launchers run from `experiments/` and write to `experiments/outputs/pa2/`. GPUs on which another
process is running are skipped.

1. Hyperparameter search on the validation stream (the 25% before the online segment, split again 20:5:75).
   Grid: learning-rate multiplier {0.5,1,2,4,8,16} x adaptation epochs {5,10,20,40} x modulation rank {2,4} x
   `z_min` {2.0,2.5,3.0}.

   ```
   python run_pa_protocol.py search --gpus 0 1 2 3
   python run_pa_protocol.py select          # writes outputs/pa2/selection.csv
   ```

2. Final runs with view switching (S), 5 seeds, on the full stream with the selected configuration:

   ```
   python run_pa_protocol.py final --gpus 0 1 2 3
   ```

3. Weighted-ensemble runs (E), same configuration, 5 seeds; and the validation-stream choice between S and E:

   ```
   python run_pa_dvablation.py --max-gpus 2 --cpu-jobs 4     # variants mod / ens / ens_full (E)
   python run_pa_modesel.py --max-gpus 2 --cpu-jobs 4        # validation runs of E
   python run_pa_modesel.py --select                         # writes outputs/pa2/mode_selection.csv
   ```

   The reported DSGA result of a (stream, horizon) pair is the `final` run if `mode_selection.csv` says
   `ftl` (S) and the `dv_ablation/ens_full` run if it says `eg` (E).

4. Default configuration (no selection): `python run_pa_default.py`.

5. Routing ablation on the controlled stream: `python run_pa_sca_ablation.py`. Route accuracy is the
   agreement of `routed_action` in each run's `routing_decisions.csv` with `oracle_actions[block]` of the
   corresponding `dataset/sca_stream_v2/seed_XX.npz`.

Single-run example (Exchange, `H=1`, view switching):

```
cd experiments
python run_pems08_prequential_multihorizon.py \
  --data ../dataset/exchange_rate/exchange_rate.csv --dataset-name Exchange --horizon 1 \
  --split proceed --lag 60 --online-blocks 8 --base-epochs 20 --seeds 5 --methods DSGA \
  --dual-view ftl --weight-mod --null-windows 128 --alpha 0.025 --graph-top-k 8 \
  --lr-scale 2 --adapt-epochs 5 --mod-rank 2 --minimum-effect-z 2.5 --output outputs/example
```

Use `--dual-view eg` for the weighted ensemble. `--route-variant` and `--no-memory` switch on the routing
ablations of the controlled stream.
