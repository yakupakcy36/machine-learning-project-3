# Wine Quality Prediction

Small learning project. The notebook `wine_quality  (1).ipynb` trains a random forest classifier.

Data: Red wine quality data (winequality-red.csv), 11 physicochemical features, predicting whether quality is good. The data files are not included in this repo, so you need to download them yourself.

## Benchmark script

`benchmark.py` compares extra trees, random forest and HistGradientBoosting with 5-fold cross-validation.
It does not use the real dataset. It runs on synthetic data generated with numpy/scikit-learn (1599 samples, 11 features), so the numbers below only compare the models on that data and say nothing about the real dataset.

## How to run

```
pip install numpy scikit-learn jupyter
jupyter notebook
python benchmark.py
```

## Results (5-fold CV, mean)

| Model | accuracy | f1 | roc_auc |
|---|---|---|---|
| Extra Trees (250 Trees) | 0.9256 | 0.812 | 0.9552 |
| HistGradientBoosting | 0.93 | 0.8308 | 0.9507 |
| Random Forest (200 Trees) | 0.9168 | 0.7886 | 0.9467 |

Full numbers are in `benchmarks/benchmark_results.json`.

## Files

- `wine_quality  (1).ipynb`: notebook
- `benchmark.py`: benchmark script
- `benchmarks/benchmark_results.json`: results
- `assets/benchmark_chart.svg`: chart of the results
