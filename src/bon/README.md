# Best-of-*n* (BoN) Inference

An explanation for [running BoN](#running-bon) is provided in the project-level README, but we also provide it here, with some additional sections, for convenience.

## Supported Pythia v1 Models

BoN experiments are supported for all Pythia v1 models. See the configuration file at `/configs/config_bon.yaml` for detailed settings.

| Base Model | RM Config | RM Path Template |
|------------|-----------|------------------|
| Pythia 70M | `rm-pythia-44m` | `models/rm-pythia-44m_seed{seed}` |
| Pythia 160M | `rm-pythia-160m` | `models/rm-pythia-160m_seed{seed}` |
| Pythia 410M | `rm-pythia-410m` | `models/rm-pythia-410m_seed{seed}` |
| Pythia 1B | `rm-pythia-1b` | `models/rm-pythia-1b_seed{seed}` |
| Pythia 1.4B | `rm-pythia-1.4b` | `models/rm-pythia-1.4b_seed{seed}` |
| Pythia 2.8B | `rm-pythia-2.8b` | `models/rm-pythia-2.8b_seed{seed}` |
| Pythia 6.9B | `rm-pythia-6.9b` | `models/rm-pythia-6.9b_seed{seed}` |
| Pythia 12B | `rm-pythia-12b` | `models/rm-pythia-12b_seed{seed}` |

LoRA variants are also available (e.g., `rm-pythia-44m-lora`, `rm-pythia-160m-lora`, etc.).

## Quick Start Examples

### Individual RM (Single Seed)
```bash
# Pythia 44M (70M base)
python src/bon/run_bon_pipeline.py models/rm-pythia-44m_seed{seed} -o bon_pythia_44m --seeds 1

# Pythia 1.4B
python src/bon/run_bon_pipeline.py models/rm-pythia-1.4b_seed{seed} -o bon_pythia_1.4b --seeds 1
```

### Ensemble BoN (5 Seeds with Mean, WCO, UWO)
```bash
# Pythia 44M (70M base)
python src/bon/run_bon_pipeline.py models/rm-pythia-44m_seed{seed} \
    -o bon_pythia_44m_ensemble --seeds 1,2,3,4,5 --ensembles

# Pythia 160M
python src/bon/run_bon_pipeline.py models/rm-pythia-160m_seed{seed} \
    -o bon_pythia_160m_ensemble --seeds 1,2,3,4,5 --ensembles

# Pythia 410M
python src/bon/run_bon_pipeline.py models/rm-pythia-410m_seed{seed} \
    -o bon_pythia_410m_ensemble --seeds 1,2,3,4,5 --ensembles

# Pythia 1B
python src/bon/run_bon_pipeline.py models/rm-pythia-1b_seed{seed} \
    -o bon_pythia_1b_ensemble --seeds 1,2,3,4,5 --ensembles

# Pythia 1.4B
python src/bon/run_bon_pipeline.py models/rm-pythia-1.4b_seed{seed} \
    -o bon_pythia_1.4b_ensemble --seeds 1,2,3,4,5 --ensembles

# Pythia 2.8B
python src/bon/run_bon_pipeline.py models/rm-pythia-2.8b_seed{seed} \
    -o bon_pythia_2.8b_ensemble --seeds 1,2,3,4,5 --ensembles

# Pythia 6.9B
python src/bon/run_bon_pipeline.py models/rm-pythia-6.9b_seed{seed} \
    -o bon_pythia_6.9b_ensemble --seeds 1,2,3,4,5 --ensembles

# Pythia 12B
python src/bon/run_bon_pipeline.py models/rm-pythia-12b_seed{seed} \
    -o bon_pythia_12b_ensemble --seeds 1,2,3,4,5 --ensembles
```

### With LoRA-trained RMs
```bash
python src/bon/run_bon_pipeline.py models/rm-pythia-44m-lora_seed{seed} \
    -o bon_pythia_44m_lora --seeds 1,2,3,4,5 --ensembles
```

## Complete Workflow

### Step 1: Train Reward Models (Multiple Seeds for Ensembles)

```bash
# Train 5 reward models with different seeds
for seed in 1 2 3 4 5; do
    accelerate launch src/reward_modeling/training/trainer_rm.py \
        --configs defaults_rm rm-pythia-44m --rng_seed $seed
done
```

This produces: `models/rm-pythia-44m_seed1`, `models/rm-pythia-44m_seed2`, etc.

### Step 2: Run BoN Sampling

```bash
python src/bon/run_bon_pipeline.py models/rm-pythia-44m_seed{seed} \
    -o bon_pythia_44m --seeds 1,2,3,4,5 --ensembles
```

### Step 3: Analyze Results

Results are saved in:
- Individual RMs: `runs/bon_pythia_44m/seed{1-5}/bon_sampled_results.json`
- Mean ensemble: `runs/bon_pythia_44m/ensemble_mean/bon_sampled_results.json`
- WCO ensemble: `runs/bon_pythia_44m/ensemble_wco/bon_sampled_results.json`
- UWO ensemble: `runs/bon_pythia_44m/ensemble_uwo-0.5/bon_sampled_results.json`

---

## Running BoN

Best-of-*n* inference (also known as re-ranking) can be performed using a base policy model (post-SFT) and one or more trained reward models, as follows:
```
python src/bon/run_bon_pipeline.py {your_reward_models_path}
```
This command is customizable with the following arguments and options:
| Argument        | Type | Required |  Description                                              | Default value              |
| --------------- | ---- | -------- | --------------------------------------------------------- | -------------------------- |
| `proxy_rm_path`             | str         | yes      | generic path to proxy (non-gold) reward models to use. This should be a string with a "{seed}" placeholder, so that multiple reward models can be retrieved, both for ensembles and general convenience. e.g. "models/rm-pythia-44m_seed{seed}"| - |
| `output_dir`                | str         | no       | name of (path to) the directory where the output model should be saved. This will go under [runs/](/runs/). | bon_sampling_{curr_time}
| `gold_gens` | str         | no       | name of (path to) BoN dataset containing at least `big_n` answers (see [dataset guide](/src/data_utils/README.md#best-of-n-bon) for details)| tlc4418/gold_labelled_gens |
| `big_n`                     | int         | no       | total number of answers to perform BoN sampling over. 'N' in the unbiased estimator formula. Usually the total number of answers in the dataset, will be used to cut down the dataset otherwise. | 12600 |
| `sample_ns`                 | str         | no       | list of indexes (the *n* in best-of-*n*) at which to perform BoN sampling. Comma-separated list of ints. | "1,2,4,8,16,32,64,128,<br>256,512,1024,2048,<br>4096,6144,8192,12500" |
| `seeds`                     | str         | no       | list of seeds corresponding to which reward models to run (will be used to fill the "{seed}" placeholder for the `proxy_rm_path`). If doing BoN for ensembles, the length of this list is also your ensemble cardinality. Comma-separated list of ints. | "1,2,3,4,5"  |
| `ensembles`                 | bool        | no       | whether to run BoN over ensembles. If set to true, BoN for the three types of ensembles in the paper (mean, WCO, UWO) will be performed in addition to the individual reward models. | True  |
| `uwo_weights`               | str         | no       | list of UWO weights to use when doing BoN sampling with the UWO ensemble (if `ensembles` is true). Results will be given for a new ensemble with each weight. Comma-separated list of floats. | "0.5" |


A help function is also provided (`--help`) which will display a condensed version of the above, including the appropriate flag names for specifying each parameter.

Our implementation uses an unbiased estimator (see paper for details) for robust and unbiased results.

The relevant result of running this command will be a "bon_sampled_results.json" results file for each run (one for each individual seed and one for each ensemble if desired). This file contains a list of dictionary entries, where each entry contains the sampled index *n*, the proxy reward model score at that *n*, and the corresponding gold reward model score at that *n*, as follows:
```
[
    {
        "n": {some_int},
        "proxy_score": {proxy_score},
        "gold_score": {gold_score}
    },

    ...
]
```
These data points can then be used to plot the BoN performance of different policies according to both proxy and gold reward model score, as a function of *n*.

## Ensemble-only BoN
If you have already run BoN sampling over the individual reward mdoels and only want to run BoN over ensembles, you can directly call the ensemble script:
```
python src/bon/run_bon_ensembles.py {your_output_dir}
```
This command is customizable with the following arguments and options:
| Argument       | Type  | Required | Description                                     | Default value      |
| -------------- | ----- | -------- | ----------------------------------------------- | ------------------ |
| `output_dir`                | str         | yes      | name of (path to) the directory where the individual reward model BoN outputs are already saved, under [/runs/](/runs/). The new ensembles results will also be saved under this directory. | -  |
| `sample_ns`                 | str         | no       | list of indexes (the *n* in best-of-*n*) at which to perform BoN sampling. Comma-separated list of ints. | "1,2,4,8,16,32,64,128,<br>256,512,1024,2048,<br>4096,6144,8192,12500" |
| `seeds`                     | str         | no       | list of seeds for the reward mdoels making up the enseble. The length of this list is also your ensemble cardinality. Comma-separated list of ints. | "1,2,3,4,5" |
| `uwo_weights`               | str         | no       | list of UWO weights to use when doing BoN sampling with the UWO ensemble. Results will be given for a new ensemble with each weight. Comma-separated list of floats. | "0.5" |
