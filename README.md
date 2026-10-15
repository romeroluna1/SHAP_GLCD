# SHAP-GLCD: Global Class Differentiation

This branch contains the implementation, examples, and evaluation
programs for SHAP-GLCD, an approach for analyzing global differential
SHAP contributions between pairs of classes.

SHAP-GLCD extends the pairwise class-differentiation perspective
of SHAP-LCD from local instances to collections of instances.

## Project structure

- `src/shap_glcd/`: shared implementation.
- `examples/tabular/`: tabular experiments.
- `examples/text/`: text classification experiments.
- `examples/image/`: image classification experiments.
- `evaluation/`: stability, consistency, and sensitivity evaluations.
- `results/`: generated figures and evaluation summaries.

## Experiments

### Tabular data

- Iris
- Vehicle
- Wine 
- Seeds
- EduDANE
- Child occupation (four classes)
- Child occupation (three classes)

### Text classification

- AG News
- 20 Newsgroups

### Image classification

- Fashion-MNIST
- BloodMNIST
- HAM10000

## Global analysis

SHAP-GLCD analyzes differential contributions between pairs of
classes using descriptive measures including:

- Mean absolute differential contribution
- Median absolute differential contribution
- Signed Mean
- Conditional Means
- Weighted Dominance

## Evaluation

Evaluation programs are available in the `evaluation/` directory.

Experiments may use balanced sampling by true class labels, with
correctly and incorrectly classified instances included.

Specific sampling parameters, perturbation settings, and random
seeds should be checked in each experiment.

## Dependencies

Different experiments use different Python environments.

For the validated four-class child occupation stability experiment:

- Python 3.10.0
- NumPy 2.2.6
- pandas 2.3.3
- scikit-learn 1.7.2
- SHAP 0.49.1
- joblib 1.5.3

Dependencies for the remaining experiments must be checked against
their corresponding execution environments.

## Artifacts and datasets

Large trained models, cached SHAP values, and datasets are not
included in this branch.

The corresponding artifact files are required for reproducing
experiments that load previously trained models or cached SHAP values.

No model retraining or SHAP recalculation is necessary when the
corresponding validated artifacts are available.

## Related repositories

- SHAP-LCD: https://github.com/romeroluna1/SHAP_LCD
- SHAP-GLCD: https://github.com/romeroluna1/SHAP_GLCD

These are independent repositories. SHAP-GLCD builds on the class-differentiation perspective introduced by SHAP-LCD.

