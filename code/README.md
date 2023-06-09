## 1. Setup

### Installing packages
```
pip install torch-geometric==2.0.1
pip install aif360==0.3.0
```


## 2. Datasets
We ran our experiments on two datasets: credit and income. All the data are present in the './dataset' folder.

## 3. Usage
### Examples
Evaluate fairness and utility performance of GCN and credit dataset

`python runmy_fairgnn.py --model gcn --dataset credit --alpha 5e-6 --beta 1 --seed 1`

The AUCROC of estimator: 0.6807
Total Individual Unfairness: 1882.07421875
Individual Unfairness for Group 1: 0.0022303774021565914
Individual Unfairness for Group 2: 0.0022345229517668486
GD: 1.0018586762967778

Evaluate fairness and utility performance of GCN and credit income

---Testing---
The AUCROC of estimator: 0.7077
Total Individual Unfairness: 2124.663330078125
Individual Unfairness for Group 1: 0.0021269216667860746
Individual Unfairness for Group 2: 0.0021285810507833958
GD: 1.0007801810584913
