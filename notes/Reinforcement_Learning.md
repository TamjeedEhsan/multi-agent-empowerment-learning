# Reinforcement Learning Fundamentals

## 1. What Is Reinforcement Learning?
**Reinforcement Learning (RL)** is a way of training an agent to make decisions by interacting with an environment and receiving feedback.

## 2. The Six Fundamental Concepts

1. **Agent:** The learner or decision-maker that chooses actions.
2. **Environment:** Everything the agent interacts with. It receives the agent's actions, changes its state, and provides observations and rewards in response.
3. **State (\(s_t\)):** A representation of the environment's situation at time \(t\).
4. **Observation (\(o_t\)):** The information the agent receives about the environment. In a fully observable environment, the observation may contain the complete state.
5. **Action (\(a_t\)):** A decision or move made by the agent that can affect the environment.
6. **Reward (\(r_t\)):** A numerical signal from the environment that evaluates the immediate outcome of an action.
7. **Policy (\(\pi\)):** The strategy the agent uses to select actions based on its available information. A policy can select actions directly or assign probabilities to possible actions.
8. **Value Function (\(V(s)\)):** An estimate of the expected cumulative discounted future reward from a given state, following a particular policy.

## 3. Total Discounted Future Reward

The **discounted return** is the total future reward accumulated from a particular time step, with future rewards weighted according to a discount factor.

\[
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
\]

### Explanation

- \(G_t\): Total discounted future reward starting at time \(t\).
- \(r_t\): Reward received at time \(t\).
- \(\gamma\): Discount factor, where \(0 \leq \gamma < 1\) is common in continuing tasks.
- \(r_{t+k}\): Reward received \(k\) steps into the future.
- \(\gamma^k\): The weight applied to a reward received \(k\) steps later.

### Why Discount Future Rewards?

The discount factor determines how strongly the agent values future rewards relative to immediate rewards.

- **Low \(\gamma\):** The agent places more emphasis on immediate rewards.
- **High \(\gamma\):** The agent places more emphasis on long-term rewards.

## 4. The Reinforcement Learning Loop

The agent and environment interact repeatedly:

1. The agent receives an observation of the environment.
2. The agent selects an action using its policy.
3. The environment transitions to a new state.
4. The environment provides a reward and a new observation.
5. The agent uses experience to improve its decision-making, depending on the learning algorithm.


# PPO — 6 Core Concepts

**1. Policy \(\pi(a\mid s)\)**  
The strategy an agent uses to choose actions based on its current state.  
*Example:* A robot chooses to move right with a 70% probability.

**2. Reward \(r_t\)**  
Immediate numerical feedback from the environment after an action.  
*Example:* +5 for moving a block closer to its target.

**3. Return \(G_t\)**  
The cumulative future reward, usually discounted by \(\gamma\).

\[
G_t=r_t+\gamma G_{t+1}
\]

**4. Advantage \(A(s,a)\)**  
Measures how much better or worse an action performs than expected.

\[
A(s,a)=Q(s,a)-V(s)
\]

Positive advantage generally encourages an action; negative advantage generally discourages it.

**5. PPO Clipped Objective**  
Compares new and old policy probabilities and discourages excessively large policy updates.

\[
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}
\]

**6. Actor and Critic**
- **Actor:** Selects actions using the policy.
- **Critic:** Estimates the expected future return from a state.

# PPO Training Loop

**1. Collect Experience**  
The agent interacts with the environment and stores transitions:

\[
(s_t,a_t,r_t,s_{t+1})
\]

**2. Calculate Advantages**  
Uses rewards and value estimates to determine whether actions performed better or worse than expected. PPO commonly uses GAE.

**3. Update the Actor**  
Uses the clipped objective and gradient-based optimization to improve the policy.

**4. Update the Critic**  
Updates its value predictions using targets derived from collected experience.

**5. Repeat**  
Collect new experience and train again until the policy improves or the training budget is reached.

### Overall Workflow

**Collect experience → Calculate advantages → Update actor and critic → Repeat.**


