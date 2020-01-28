[**Deep Reinforcement Learning**](https://github.com/RedLeader962/LectureDirigeDRLimplementation/tree/master)

![TaxonomySoftActorCritic](./visual/TaxonomySoftActorCritic_mod.png) 

# :: Soft Actor-Critic


Soft Actor-Critic (_SAC_) is an off-policy algorithm based on the _Maximum Entropy_ \textit{Reinforcement Learning} framework.
The main idea behind \textit{Maximum Entropy RL} is to frame the decision-making problem as a graphical model from top to bottom and then solve it using tools borrowed from the field of \textit{Probabilistic Graphical Model}. Under this framework, a learning agent seeks to maximize both the return and the entropy simultaneously.
This approach benefit \textit{Deep Reinforcement Learning} algorithm by giving them the capacity to consider and learn many alternate paths leading to an optimal goal
and the capacity to learn how to act optimally despite adverse circumstances.\newline

%    This approach benefit \textit{Deep Reinforcement Learning} algorithm by giving them:
%    \begin{itemize}[ labelindent=0.5em, style=multiline,leftmargin=1.5em, itemsep=-1pt,]
%        \item the capacity to consider and learn many alternate path leading to a optimale goal;
%        \item and the capacity to learn how to act optimaly despite adverse circumstances;
%    \end{itemize}

Since \textit{SAC} is an off-policy algorithm, then it has the ability to train on samples coming from a different policy.
What is particular though is that contrary to other off-policy algortihm, it's stable. This mean that the algorithm is much less picky in term of hyperparameter tuning.\newline

\textit{SAC}~\cite{Haarnoja2018a} is curently \textbf{the state of the art} \textit{Deep Reinforcement Learning} algorithm together with Twin Delayed Deep Deterministic policy gradient (\textit{TD3})~\cite{DBLP:journals/corr/abs-1802-09477}.\newline


The learning curve of the \textit{Maximum Entropy RL} framework is quite steep due to it's depth and to how much it re-think the RL problem. It was definitavely required in order to understand how \textit{SAC} work.
Tackling the applied part was arguably the most difficult project I did to date, both in term of component to implement \& silent bug dificulties.
Never the less I'm particularly proud of the result.\newline

You can find my implementaion at \href{https://github.com/RedLeader962/LectureDirigeDRLimplementation}{https://github.com/RedLeader962/LectureDirigeDRLimplementation}

\subsection{Reading material:}
\begin{itemize}[ labelindent=0.5em, style=multiline,leftmargin=1.5em, itemsep=-1pt,]
    \item Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor~\cite{Haarnoja2018a}
    \item Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review~\cite{Levine2018}
    \item Soft Actor-Critic Algorithms and Applications~\cite{Haarnoja2018}
    \item Reinforcement Learning with Deep Energy-Based Policies~\cite{Haarnoja2017}
    \item Deterministic Policy Gradient Algorithms~\cite{Silver2014}
    \item Reinforcement learning: An introduction~\cite{Sutton1394}
\end{itemize}


I've also complemented my reading with the following resources:
\begin{itemize}[ labelindent=0.5em, style=multiline,leftmargin=1.5em, itemsep=-1pt,]  % option: itemsep=-2pt, [before=\small, labelindent=1em, style=multiline, itemsep=-1pt]
    %        [before=\footnotesize,leftmargin=0.5em, ] % option: labelindent=1em, leftmargin=2em, before=\footnotesize, font=\small, itemsep=-2pt, style=multiline, label=\(\circ\),
    \item \href{http://rail.eecs.berkeley.edu/deeprlcourse-fa18/}{CS 294--112 \textit{Deep Reinforcement Learning}}: lectures 14-15 by Sergey Levine from University Berkeley;
    \item \href{https://spinningup.openai.com/en/latest/algorithms/sac.html}{\textit{OpenAI: Spinning Up: \textit{Soft Actor-Critic}}}, by Josh Achiam;
    \item and also \href{https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#sac}{\textit{Lil' Log blog:Policy Gradient Algorithms}}  by Lilian Weng, research intern at \textit{OpenAI};
\end{itemize}


___

Advantage Actor-Critic method are close cousin of Policy Gradient class algorithm. The difference is that they use two neural networks instead of one: the **actor** who has the responsibility of finding the best action given a observation and the **critic** who has the responsibility of assessing if the actor does a good job.

**The two main goals of this essay** were to first, get a deeper understanding of Actor-Critic method theoric aspect and second, to acquire a practical understanding of it’s beavior, limitation and requirement in order to work. In order to reach this second goal, I felt it was nescessary to implement multiple design & architectural variation commonly found in the litterature.
  
With this in mind, I’ve focused on the following practical aspect:
- **Algorithm type**: batch vs online;
- **Computation graph**: split network vs split network (with shared lower layer) vs shared network;
- **Critic target**: Monte-Carlo vs bootstrap estimate;
- **Math computation**: element wise vs graph computed;
- Various **Data collection** strategy;
        
        
In parallel, I writen a second essay _A reflexion on design, architecture and implementation details_ where I go further in my study of somme aspect of DRL algortihm from a software engineering perspective applied to research by covering question like:

**Does implementation details realy matters? Which one does, when & why?**


I've also complemented my reading with the following ressources:

- The classic book [Reinforcement learning: An introduction 2nd ed.](http://incompleteideas.net/book/RLbook2018.pdf) by Sutton & Barto (ed MIT Press)
- [CS 294--112 Deep Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse-fa18/): lecture on Policy Gradient and Actor-Critic by Sergey Levine from University Berkeley;
- [OpenAI: Spinning Up: Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html), by Josh Achiam;
- and [Lil' Log blog:Policy Gradient Algorithms](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html) by Lilian Weng, research intern at OpenAI
- [Asynchronous Methods for Deep Reinforcement Learning.](https://arxiv.org/abs/1602.01783) by Mnih et al.  
- [Reinforcement learning that matters](https://arxiv.org/abs/1709.06560) by Henderson et al. 
- [TD or not TD: Analyzing the Role of Temporal Differencing in Deep Reinforcement Learning ](http://arxiv.org/abs/1806.01175) by Amiranashvili, Dosovitskiy, Koltun & Brox 
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) by Schulman, Moritz, Levine, Jordan & Abbeel

---
Download the essay pdf:
- [Deep Reinforcement Learning – Actor-Critic](https://github.com/RedLeader962/LectureDirigeDRLimplementation/raw/master/TP_actor_critic_LucCoupal_v1-1.pdf) 
- [A reflexion on design, architecture and implementation details](https://github.com/RedLeader962/LectureDirigeDRLimplementation/raw/master/Reflexion_on_design_and_architecture_LucCoupal_v1-1.pdf) 



Watch [recorded agent](../../video) 

---

### The Actor-Critic implementations:
Note: You can check explanation on how to use the package by using the `--help` flag

#### To watch the trained algorithm 

```bash
cd DRLimplementation
python -m ActorCritic --play[Lunar or Cartpole] [--record] [--play_for]=max trajectories (default=10) 
```

#### To execute the training loop
```bash
cd DRLimplementation
python -m ActorCritic --trainExperimentSpecification [--rerun] [--renderTraining] 
```
Choose `--trainExperimentSpecification` between the following:
- **_CartPole-v0_ environment:**
    - `--trainSplitMC`: Train a Batch Actor-Critic agent with Monte Carlo TD target
    - `--trainSplitBootstrap`: Train a Batch Actor-Critic agent with bootstrap estimate TD target
    - `--trainSharedBootstrap`: Train a Batch Actor-Critic agent with shared network
    - `--trainOnlineSplit`: Train a Online Actor-Critic agent with split network
    - `--trainOnlineSplit3layer`: Train a Online Actor-Critic agent with split network
    - `--trainOnlineShared3layer`: Train a Online Actor-Critic agent with Shared network
    - `--trainOnlineSplitTwoInputAdvantage`: Train a Online Actor-Critic agent with split Two input Advantage network
- **_LunarLander-v2_ environment:**
    - `--trainOnlineLunarLander`: Train on LunarLander a Online Actor-Critic agent with split Two input Advantage network
    - `--trainBatchLunarLander`: Train on LunarLander a Batch Actor-Critic agent 


#### To navigate trough the computation graph in TensorBoard
```bash
cd DRLimplementation
tensorboard --logdir=ActorCritic/graph
```

![Trained agent in action](../../video/Batch_ActorCriticBatch-AAC-Split-nn_1.gif)


---
