# Learnable Reward Function

The goal of this project is to explore an approach to learn a smooth reward function out of a sparse reward function in the context of Reinforcement Learning. This learnt reward function should make learning converge faster.

## Main goals

* Learn a smooth reward function given a sparse reward function.
* Use a learnt smooth reward function to train an agent to meet the goal of the sparse reward function.
* Compare the learnt smooth reward function against the sparse reward function (and ideally show that the learnt reward function makes agent learning converge faster).


## Status

### Done

* A reasonable reward function is learnt for the CartPole environment. The reward values assigned to states can be observed on the heatmap plot.

### Issues

* The learnt reward function doesn't seem to let a policy learn sufficiently well.
* Policy learning using stage-based difficulty adjustment doesn't necessarily adjust difficulty according to agent's abilities. Sometimes it increases difficulty too fast and the agent is unable to obtain enough possitive reward. This usually leads the agent not to improve in the right direction, potentially getting stuck in a bad solution.

### Other observations

* For some reason, the learnt reward function only produces negative values. This could be a cause of slow learning.


## Next steps

### Prioritary

* Implementation: split components for better readibility and more flexibility to updates. The following components should be sequential: regular environment, transformation to sparse reward environment, learnt reward network and weighted average, agent.
* Replace stage-based difficulty adjustment with performance-based difficulty adjustment. Test performance.
* Replace stochastic gradient descent with mini-batch gradient descent. Test performance.

### Optional

* During reward function training, we could try interleaving between training the agent and the reward function. When one has its weights fixed, the other has them updated. This may increase stability while learning the reward function.