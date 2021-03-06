[**Deep Reinforcement Learning**](https://github.com/RedLeader962/LectureDirigeDRLimplementation/tree/master)

![TaxonomyPolicyGradient](./visual/TaxonomyPGgithub.png) 

# :: Basic policy gradient
Policy gradient is a on-policy method which seek to directly optimize the policy  by using sampled trajectories as weight. Those weights will then be used to indicate how good the policy performed. Based on that knowledge, the algorithm updates the parameters of his policy to make action leading to similar good trajectories more likely and similar bad trajectories less likely. In the case of Deep Reinforcement Learning, the policy parameter is a neural net. For this essay, I've studied and implemented the basic version of policy gradient also known as REINFORCE. I've also complemented my reading with the following ressources:

- [CS 294--112 Deep Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse-fa18/): lecture 4, 5 and 9 by Sergey Levine from University Berkeley;
- [OpenAI: Spinning Up: Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html), by Josh Achiam;
- and [Lil' Log blog:Policy Gradient Algorithms](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html) by Lilian Weng, research intern at OpenAI

---
[Download the essay pdf](https://github.com/RedLeader962/LectureDirigeDRLimplementation/raw/master/Policy_gradient_LucCoupal_v1-1.pdf) 



Watch [recorded agent](../../video/REINFORCE_video) 

---

### The REINFORCE implementation:
Note: You can check explanation on how to use the package by using the `--help` flag

##### To watch the trained algorithm 

```bash
cd DRLimplementation
python -m BasicPolicyGradient [--record] [--play_for]=max trajectories (default=10)  
```

##### To execute the training loop
```bash
cd DRLimplementation
python -m BasicPolicyGradient --train
```


##### To navigate trough the computation graph in TensorBoard
```bash
tensorboard --logdir=DRLimplementation/BasicPolicyGradient/graph/runs
```

![Trained agent in action](../../video/REINFORCE_video/REINFORCE_agent_cartpole_2.gif)


---
