# Fairness Dataset Experiments

This repository contains the code to conduct classification experiments on fairness datasets using a L2-regularized logistic regression model. This includes the experiments on fairness datasets in the paper "Characterizing the Representation Disparity of Differential Privacy" by Gardner, Morgenstern, and Oh (2021).

# Usage

Configure environment by running: `pip3 install -r requirements.txt`.

For our experiments, we use Python 3.6.7 and a laptop CPU. Running each training iteration takes only a few minutes.

The main script used to run experiments is `train_l2lr.py`. Example commands to train various models are included in the header of that file. One example:

``` 
python3 train_l2lr.py --learning_rate 0.01 --epochs 1000 --dataset german --niters 50 \
    --sensitive_attribute sex \
    --majority_attribute_label male \
    --minority_attribute_label female \
    --batch_size 64 \
    --dp --dp_l2_norm_clip 10.0 --dp_noise_multiplier 0.4
```

# Datasets

The following datasets are required for the experiments in this repo:

* German Credit ([link](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)))  
* COMPAS ([link](https://github.com/propublica/compas-analysis))
* Adult Income ([link](https://archive.ics.uci.edu/ml/datasets/adult))

We note that our preprocessing of the COMPAS dataset follows the open-source code of [1], which was accessed on [codalab](https://worksheets.codalab.org/bundles/0x2074cd3a10934e81accd6db433430ce8).

[1] Khani, F., Raghunathan, A., and Liang, P. Maximum weighted loss discrepancy. arXiv preprintarXiv:1906.03518, 2019.