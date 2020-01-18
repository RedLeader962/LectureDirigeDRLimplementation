# coding=utf-8

from blocAndTools.buildingbloc import GymPlayground, gym_environment_reward_assesment
from blocAndTools.rl_vocabulary import rl_name

vocab = rl_name()

playgroundLunarLanderContinuous = GymPlayground('LunarLanderContinuous-v2')
playgroundMountainCarContinuous = GymPlayground('MountainCarContinuous-v0')
playgroundBipedalWalkerContinuous = GymPlayground('BipedalWalker-v2')
playgroundBipedalWalkerHardcoreContinuous = GymPlayground('BipedalWalkerHardcore-v2')
playgroundPendulum = GymPlayground('Pendulum-v0')
playgroundCartPole = GymPlayground('CartPole-v1')

""" Average reward over 5000 sample in environment:
        CartPole: 							1.0
        Pendulum: 							-6.194059069237778
        MountainCarContinuous: 				-0.03379479089447222
        LunarLanderContinuous: 				-1.8451261821262512
        BipedalWalkerContinuous: 			-0.09389189287451959
        BipedalWalkerHardcoreContinuous: 	-0.21095661603674906
        
    Average reward over 10000 sample in environment:
        CartPole: 							1.0
        Pendulum: 							-6.392986420889935
        MountainCarContinuous: 				-0.033452527858401936
        LunarLanderContinuous: 				-2.1619796013042705
        BipedalWalkerContinuous: 			-0.09298591752622176
        BipedalWalkerHardcoreContinuous: 	-0.056385564141322744
    
    Average reward over 30000 sample in environment:
        CartPole: 							1.0
        Pendulum: 							-5.5914760489795725
        MountainCarContinuous: 				-0.029942065782727064
        LunarLanderContinuous: 				-2.0475705532632373
        BipedalWalkerContinuous: 			-0.06094746792211208
        BipedalWalkerHardcoreContinuous: 	-0.05619588020553646
        
    Average reward over 100000 sample in environment:
        CartPole: 							1.0
        Pendulum: 							-5.687553812910036
        MountainCarContinuous: 				-0.03120409531395761
        LunarLanderContinuous: 				-2.0004798743377448
        BipedalWalkerContinuous: 			-0.05886533021377718
        BipedalWalkerHardcoreContinuous: 	-0.07178085108315299
"""

SAMPLE_SIZE = int(1e5)
# SAMPLE_SIZE = 5000
LunarLanderContinuousRewardAssesement = gym_environment_reward_assesment(
    playgroundLunarLanderContinuous.env.env, sample_size=SAMPLE_SIZE)
MountainCarContinuousRewardAssesement = gym_environment_reward_assesment(
    playgroundMountainCarContinuous.env.env, sample_size=SAMPLE_SIZE)
BipedalWalkerContinuousRewardAssesement = gym_environment_reward_assesment(
    playgroundBipedalWalkerContinuous.env.env, sample_size=SAMPLE_SIZE)
BipedalWalkerHardcoreContinuousRewardAssesement = gym_environment_reward_assesment(
    playgroundBipedalWalkerHardcoreContinuous.env.env, sample_size=SAMPLE_SIZE)
PendulumRewardAssesement = gym_environment_reward_assesment(
    playgroundPendulum.env.env, sample_size=SAMPLE_SIZE)
CartPoleRewardAssesement = gym_environment_reward_assesment(
    playgroundCartPole.env.env, sample_size=SAMPLE_SIZE)

print('Average reward over {} sample in environment:\n'.format(SAMPLE_SIZE),
      '\tCartPole: \t\t\t\t\t\t\t{}\n'.format(CartPoleRewardAssesement),
      '\tPendulum: \t\t\t\t\t\t\t{}\n'.format(PendulumRewardAssesement),
      '\tMountainCarContinuous: \t\t\t\t{}\n'.format(MountainCarContinuousRewardAssesement),
      '\tLunarLanderContinuous: \t\t\t\t{}\n'.format(LunarLanderContinuousRewardAssesement),
      '\tBipedalWalkerContinuous: \t\t\t{}\n'.format(BipedalWalkerContinuousRewardAssesement),
      '\tBipedalWalkerHardcoreContinuous: \t{}\n'.format(BipedalWalkerHardcoreContinuousRewardAssesement),
      )
