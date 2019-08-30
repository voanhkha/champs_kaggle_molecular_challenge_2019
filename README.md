# champs_kaggle_molecular_challenge_kaggle_2019

The sample code of the team rank 10/1874 [Kha | Zidmie | Josh | Kyle | Akira] in the "Predicting Molecular Properties 2019" competition on Kaggle.

For more information, see
[Leaderboard](https://www.kaggle.com/c/champs-scalar-coupling/leaderboard); [Solution Write-up](https://www.kaggle.com/c/champs-scalar-coupling/discussion/106271#latest-612843); [Simple GCN Kernel](https://www.kaggle.com/joshxsarah/custom-gcn-10th-place-solution); and [Lightweight SchNet kernel](https://www.kaggle.com/petersk20/schnet-10th-place-solution)

![Final Leaderboard](https://github.com/voanhkha/champs_kaggle_molecular_challenge_2019/blob/master/Molecule_Leaderboard.png)

The main successful methods during the competition:
1. Feature permutation importance, which worked well with handcrafted features used in tree-based models.
2. SchNet for molecular data, which is an end-to-end graph architecture.
The training and predicting scripts require a sample dataset for 1 coupling type (1JHN), which can be found [here](https://drive.google.com/drive/folders/13VxPs5N8JcGci3sGd9PM7XlirtgTBZ_C?usp=sharing).


Kha Vo, 30th August 2019.
