## GFIF: Individual  Fairness under Group Fairness Constraints in Graph Neural Networks for Node Classification Task
## 1. Setup
```
cd code
```
### Installing packages
Please run the following commands to install necessary packages.
For more details on Pytorch Geometric please refer to install the PyTorch Geometric packages following the instructions from [here.](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

```
conda create --name ifgf python==3.7.11
conda activate ifgf
conda install pytorch==1.10.0 cudatoolkit=11.7 -c pytorch -c conda-forge
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
pip install torch-geometric==2.0.1
pip install aif360==0.3.0
```

## 2. Datasets
We ran our experiments on two datasets: credit and income. All the data are present in the './dataset' folder.

## 3. Usage
### Examples
Evaluate fairness and utility performance of GCN and credit dataset

`python runmy_fairgnn.py --model gcn --dataset credit --alpha 5e-6 --beta 1 --seed 1`

<p align="left"><i>
  The AUCROC of estimator: 0.6807<br/>
  Total Individual Unfairness: 1882.07421875<br/>
  Individual Unfairness for Group 1: 0.0022303774021565914<br/>
  Individual Unfairness for Group 2: 0.0022345229517668486<br/>
  GD: 1.0018586762967778<br/>
</i></p> 

Evaluate fairness and utility performance of GCN and credit income

---Testing---
<p align="left"><i>
  The AUCROC of estimator: 0.7077<br/>
  Total Individual Unfairness: 2124.663330078125<br/>
  Individual Unfairness for Group 1: 0.0021269216667860746<br/>
  Individual Unfairness for Group 2: 0.0021285810507833958<br/>
  GD: 1.0007801810584913<br/>
</i></p> 

Model can be model = {GIN, GCN}

### Baselines
We adapted base implementation from https://github.com/weihaosong/GUIDE.git and built our framework upon that. Baselines can be run from same link.
