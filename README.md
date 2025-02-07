# CHAIN
The code implementation of Churn Approximation Reduction (CHAIN) for NeurIPS 2024 paper "Improving Deep Reinforcement Learning by Reducing the Chain Effect of Value and Policy Churn"


Tidying up ... Coming soon

---

### Running Environment Setups:

Since our codes are wirtten upon existing code bases, one can run our code by following the installation of corresponding environments. 

We also provide the Dockerfile we used in ```./dockerfiles``` for reference. In practice, we mainly run with Apptainer (by building docker images first and converting them to Apptainer images).

For all our code implementations, we implement based on the corresponding code base with a slight re-organization of code structure (e.g., from single-file implementation to a bit modular). One can compare the scripts for the baseline method and its CHAIN version to see the difference.

#### MinAtar
For our experiments on MinAtar, we implement based on the original code provided in the official repo https://github.com/kenjyoung/MinAtar.
We recommend to follow the installation guidance of MinAtar env and then run our code provided in ```./minatar```.

#### MuJoCo


#### DMC Suite

For our experiments on DMC Suite, we implement based on the original code provided in the official repo https://github.com/vwxyzjn/cleanrl.
We recommend to follow the installation guidance of DMC env in CleanRL and then run our code provided in ```./dmc```.


#### CORL