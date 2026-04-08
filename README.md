# FGeo-ISRL

Geometric problem-solving has always been a great challenge in the field of deductive
reasoning and artificial intelligence. Symmetry is a defining characteristic of geometric
shapes and properties. Consequently, the application of symmetry principles to geometric
reasoning arises as a natural choice. To address the efficiency degradation and limited
generalization, we propose FGeo-ISRL, a neural-symbolic inverse search framework whose
core is the synergistic integration of a task-fine-tuned large language model and Monte
Carlo Tree Search. Under the formal framework of FormalGeo, geometric theorems are
iteratively applied starting from the given conditions and the target conclusion, in order
to infer the necessary supporting premises. The large language model simultaneously
serves as a policy network and a value network, guiding theorem application decisions
and evaluating intermediate proof states, whereas the Monte Carlo Tree Search performs
structured exploration over the state space, both training for policy refinement and inference
for online search. The reinforcement learning agent is trained with a hybrid reward scheme,
combining immediate feedback from the value difference and a sparse success reward.
