# Experiment Data

We release the experiment data to facilitate future research. Each file is a dictionary keyed by `(task_name/domain_name/all, method_name)` tuples. Each entry contains an numpy array of shape `(30, 12)` (except for RLPD where the shape is `(10, 12)`). The array stores the success rate at a regular interval of 50K training steps for 12 seeds.

See [sanity-check.ipynb](sanity-check.ipynb) for some quick examples for visualizing and plotting our experiment data.

## Update-05-09:
There was a bug in the QSM implementation. We have fixed the issue, re-ran all the experiments affected, and updated the paper accordingly. The data in the repo has also been updated. For the old QSM data, please check out an older commit (e.g., [cc786f](https://github.com/ColinQiyangLi/qam/commit/f93efd22520a79f6819f458dffe9c075eacc786f)).

After the fix the offline performance went down a bit (from 42 to 39) and the online performance went up a bit. See plots for all individual tasks here: [../assets/qsm-fix-ablation.pdf](../assets/qsm-fix-ablation.pdf)
