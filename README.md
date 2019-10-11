Repository for the course IFT-7014: Directed reading on
# Deep Reinforcement Learning 

by **Luc Coupal**,
Université Laval,
Montréal, QC, Canada,
[Luc.Coupal.1@uLaval.ca](Luc.Coupal.1@uLaval.ca) 

#### Under the supervision of:

**Professor Brahim Chaib-draa**,
Directeur du programme de baccalauréat en génie logiciel de l'Université Laval,
Québec, QC, Canada,
[Brahim.Chaib-draa@ift.ulaval.ca](Brahim.Chaib-draa@ift.ulaval.ca)

---
### Essay on:
- **[Basic policy gradient](DRLimplementation/DRLTP1PolicyGradient)**
- Actor-Critic
- Maximum Entropy DRL
- Inverse DRL
- Meta-DRL

---
    
### Install instruction:
1) **Create & activate a new virtual environment** (I recommand using [conda](https://www.anaconda.com/distribution/), ... it's a walk in the park)
    ```bash
    conda create --name myNewVirtualEnvironmentName python=3.7
    conda activate myNewVirtualEnvironmentName
    ```
2) **Clone** the GitHub repository & **install dependencies**:
    ```bash
    git clone https://github.com/RedLeader962/LectureDirigeDRLimplementation.git
    cd LectureDirigeDRLimplementation
    pip install -e .
    ```
    This will automaticaly install those **dependencies** in `myNewVirtualEnvironmentName` :

        'gym>=0.14.0'
        'tensorflow>=1.14.0,<2.0',
        'matplotlib>=3.1.0',
        'numpy>=1.16.4',
        'seaborn>=0.9.0',
        'pytest',
    
3) **Enjoy** DRL script

![Trained agent in action](video/REINFORCE_agent_cartpole_2.gif)

---

