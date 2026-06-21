# Citations and Acknowledgments

## Course

- EECS 590 (Reinforcement Learning), Spring 2026. Slide decks `lect1.pdf` and `lec2.pdf`–`lec6.pdf` provided by the instructor were used as the primary reference for Markov decision processes, value iteration, Q-learning, function approximation, and policy gradients. The Version-3 project prompt itself (`healthy_lifestyle_rl_agent_prompt.md`) was provided by the instructor.

## Reinforcement learning theory

- Sutton, R. S. and Barto, A. G. *Reinforcement Learning: An Introduction*, second edition. MIT Press, 2018. — Core MDP formulation (Chapter 3), Bellman equations, Q-learning derivation (Chapter 6), and function approximation with neural networks (Chapter 9). The DQNConfig's `gamma`, `epsilon_start/end`, and `target_update_freq` fields directly mirror the variables discussed in these chapters.

## Algorithms

- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... and Hassabis, D. "Human-level control through deep reinforcement learning." *Nature* 518, 529–533 (2015). https://doi.org/10.1038/nature14236 — The primary DQN reference. The target network, experience replay, and Huber (smooth L1) loss used in `src/lifestyle_rl/agents/dqn_agent.py` trace directly to this paper.

- Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., and Riedmiller, M. "Playing Atari with Deep Reinforcement Learning." arXiv:1312.5602 (2013). — The earlier DQN workshop paper that first introduced the experience replay buffer construction and epsilon-greedy schedule used in `replay_buffer.py` and the `epsilon()` method of `DQNAgent`.

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. "Proximal Policy Optimization Algorithms." arXiv:1707.06347 (2017). — The PPO algorithm wrapped in `src/lifestyle_rl/agents/ppo_agent.py` via Stable-Baselines3. The clipped surrogate objective and `clip_range=0.2` default come from this paper.

- Schulman, J., Moritz, P., Levine, S., Jordan, M., and Abbeel, P. "High-Dimensional Continuous Control Using Generalized Advantage Estimation." arXiv:1506.02438 (2015). — The Generalized Advantage Estimation (GAE) used internally by PPO (SB3's default `gae_lambda=0.95`). The SB3 PPO wrapper automatically uses GAE; the original formulation is from this paper.

## Libraries

- Towers, M., Kwiatkowski, A., Terry, J., Balis, J. U., De Cola, G., Deleu, T., ... and Younis, O. M. "Gymnasium: A Standard Interface for Reinforcement Learning Environments." arXiv:2407.17032, 2023. https://gymnasium.farama.org — The API (`reset()`, `step()`, `action_space`, `observation_space`) that `HealthyLifestyleEnv` implements.

- Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., and Zaremba, W. "OpenAI Gym." arXiv:1606.01540 (2016). — The predecessor to Gymnasium; the `gym.Env` interface design is from this work and Gymnasium inherits it. Mentioned in the README's citation section.

- Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., and Dormann, N. "Stable-Baselines3: Reliable Reinforcement Learning Implementations." *Journal of Machine Learning Research* 22(268), 1–8, 2021. https://stable-baselines3.readthedocs.io — Used for the PPO agent in `src/lifestyle_rl/agents/ppo_agent.py`. The `MlpPolicy`, `n_steps=512`, and `ent_coef=0.01` hyperparameters trace to SB3's PPO implementation.

- Akiba, T., Sano, S., Yanase, T., Ohta, T., and Koyama, M. "Optuna: A Next-generation Hyperparameter Optimization Framework." In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2623–2631, 2019. https://optuna.org — Used in `scripts/tune_bayesian.py`. The TPE (Tree-structured Parzen Estimator) sampler is Optuna's default and is the "Bayesian" component of the Bayesian hyperparameter search.

- Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... and Chintala, S. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." In *Advances in Neural Information Processing Systems* 32, 8026–8037, 2019. — Framework used for the custom DQN (QNetwork, Adam optimizer, gradient clipping).

- Harris, C. R. et al. "Array programming with NumPy." *Nature* 585, 357–362 (2020). — Used throughout the codebase for numeric arrays, seeded RNG (`np.random.default_rng`), and the stochastic components of the transition model.

- Pandas, Matplotlib, PyYAML, tqdm, pytest — used as standard scientific-Python tooling for metrics logging, plotting, config loading, progress bars, and testing respectively.

## Equations and formulas used in the simulator

The simulator is a *teaching* simulator, not a clinical one. The following two public formulas are used so the simulator's numbers are at least grounded:

- Mifflin, M. D., St Jeor, S. T., Hill, L. A., Scott, B. J., Daugherty, S. A., and Koh, Y. O. "A new predictive equation for resting energy expenditure in healthy individuals." *American Journal of Clinical Nutrition* 51(2), 241–247 (1990). — The Mifflin-St Jeor equation used in `UserProfile.basal_metabolic_rate()` to initialize the calorie target. No medical recommendations are made on the basis of this estimate; it is solely a number inside the env initialization.

- The widely cited 7700 kcal ≈ 1 kg constant is used in `TransitionModel.step` for the slow-weight-evolution term (`KCAL_PER_KG = 7700.0`). This figure appears across sports-science and dietetics literature as a rule-of-thumb. Again, this is a simulation convenience, not a clinical claim.

## Citation traceability table

The following table maps each major claim or implementation choice in `docs/capstone-report.md` to its supporting reference. This is provided to assist with traceability and to make clear that no claim is fabricated.

| Claim / Implementation | Source |
|---|---|
| MDP formulation (states, actions, transitions, rewards, discount) | Sutton & Barto (2018), Ch. 3 |
| Bellman target: `Q(s,a) ← r + γ·max_a' Q(s',a')` | Sutton & Barto (2018), Ch. 6; Mnih et al. Nature (2015) |
| Experience replay and target network stabilize DQN training | Mnih et al. Nature (2015); Mnih et al. arXiv:1312.5602 (2013) |
| Huber (smooth L1) loss for DQN | Mnih et al. Nature (2015) |
| Epsilon-greedy exploration schedule | Sutton & Barto (2018), Ch. 2; Mnih et al. arXiv:1312.5602 (2013) |
| PPO clipped surrogate objective, `clip_range=0.2` | Schulman et al. arXiv:1707.06347 (2017) |
| GAE (λ=0.95) inside PPO | Schulman et al. arXiv:1506.02438 (2015) |
| TPE sampler for Bayesian hyperparameter search (Optuna) | Akiba et al. KDD (2019) |
| `gym.Env` API (`reset`, `step`, spaces) | Towers et al. arXiv:2407.17032 (2023); Brockman et al. (2016) |
| Mifflin-St Jeor BMR for calorie initialization | Mifflin et al. AJCN (1990) |
| 7700 kcal/kg energy-balance constant | Mifflin et al. AJCN (1990); widely cited in sports-science literature |

## People and tools that helped

- Course instructor and TAs for EECS 590 — feedback on V1 and V2 capstone submissions shaped the scope, the safety framing, and the decision to write DQN from scratch while using SB3 for PPO.
- Anthropic Claude (used as a pair-programming and drafting assistant) — see the README's "Tools / chatbot disclosure" section for the scope of that assistance. All design and algorithmic decisions are the author's; AI-suggested code was reviewed and tested before being kept.
- Classmates in the EECS 590 cohort, for project-update discussion sessions where the single-agent framing was sharpened against multi-agent and robotics alternatives.
