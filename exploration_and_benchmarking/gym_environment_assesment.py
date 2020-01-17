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

""" Average reward over 1000 sample in environment:
            CartPole: 							0.014
            Pendulum: 							-7.669450253453739
            MountainCarContinuous: 				-0.03374852700103509
            LunarLanderContinuous: 				-90.00947448084858
            BipedalWalkerContinuous: 			-94.30254244639558
            BipedalWalkerHardcoreContinuous: 	-96.2067754529273
            
    Average reward over 5000 sample in environment:
            CartPole: 							0.0036
            Pendulum: 							-4.863037248615252
            MountainCarContinuous: 				-0.033343435949857496
            LunarLanderContinuous: 				-98.78142879872294
            BipedalWalkerContinuous: 			-98.28313092135646
            BipedalWalkerHardcoreContinuous: 	-98.84284579807118
"""

# SAMPLE_SIZE = int(1e5)
SAMPLE_SIZE = 5000
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
