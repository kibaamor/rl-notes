# 强化学习笔记

[TOC]

--------

## DQN中done参数对模型训练的影响

### 源代码

[dqn.py](./dqn_done_test/dqn.py)

关键代码

```python
q_pred = self.model(s0).gather(1, a0).squeeze()
with torch.no_grad():
    if self.use_dbqn:
        acts = self.model(s1).max(1)[1].unsqueeze(1)
        q_target = self.target_model(s1).gather(1, acts).squeeze(1)
    else:
        q_target = self.target_model(s1).max(1)[0]

    if self.use_done:
        q_target = r1 + self.gamma * (1 - d1) * q_target
    else:
        q_target = r1 + self.gamma * q_target
loss = self.loss_func(q_pred, q_target)
```

### 测试

测试环境主要参数

|       参数名       |   参数值    |
| :----------------: | :---------: |
|     gym环境名      | CartPole-v0 |
|     测试回合数     |   10000次   |
| Hidden Layer 大小  |     32      |
| Hidden Layer 层数  |      2      |
| Replay Memory 大小 |    10000    |
|  学习 Batch 大小   |     128     |
|       学习率       |    1e-3     |
|       Gamma        |    0.99     |

1. DQN使用done参数

    ```bash
    python -m dqn_done_test.dqn --exp-name=with-done-test --use-done=True
    ```

2. DQN不使用done参数

    ```bash
    python -m dqn_done_test.dqn --exp-name=without-done-test --use-done=False
    ```

3. DBQN(Double-Q Learning)使用done参数

    ```bash
    python -m dqn_done_test.dqn --exp-name=dbqn-with-done-test --use-done=True --use-dbqn=True
    ```

4. DBQN(Double-Q Learning)不使用done参数

    ```bash
    python -m dqn_done_test.dqn --exp-name=dbqn-without-done-test --use-done=False --use-dbqn=True
    ```

### 结果对比

1. DQN使用done参数与否（**橙色**使用done参数，**蓝色**未使用done参数）

    ![dqn_done_test](https://cdn.jsdelivr.net/gh/KibaAmor/rl-notes/dqn_done_test/dqn_done_test.png)

    > tips: 使用done参数时，测试耗时53分钟，不使用done参数时，测试耗时3分钟。

2. DBQN(Double-Q Learning)使用done参数与否（**浅蓝色**使用done参数，**红色**未使用done参数）

    ![dbqn_done_test](https://cdn.jsdelivr.net/gh/KibaAmor/rl-notes/dqn_done_test/dbqn_done_test.png)

    > tips: 使用done参数时，测试耗时54分钟，不使用done参数时，测试耗时3分钟。

### 结果分析

在DQN算法中，done参数还是需要使用的。如果自己实现的DQN算法，在训练时难以收敛，可以看看done参数是否参与了训练。网上关于RL算法的实现有很多（有一些没有target model，有一些不用replay memory），质量参差不齐，参考时需谨慎。

### Extra: DQN对比DBQN(Double-Q Learning)

![dqn_vs_dbqn](https://cdn.jsdelivr.net/gh/KibaAmor/rl-notes/dqn_done_test/dqn_vs_dbqn.png)

> tips: 橙色DQN，浅蓝色DBQN。

--------

## 实现PG算法时，注意softmax的dim参数

PG算法中，Model的最后一个activation是softmax，softmax函数需要指定其生效的dim。

```python
class Model(nn.Module):
    ...

    def forward(self, obs):
        obs = torch.tanh(self.fc1(obs))
        action_prob = F.softmax(self.fc2(obs), dim=1)
        return action_prob
```

在learn时需要根据一系列的obs来计算一系列的prob，所以dim的参数值应当为1，如上面的代码片段。那么在predict时就需要将单个的obs变换成一个二维数组再进行运算。

```python
class Agent:
    ...

    def predict(self, obs):
        obs = torch.FloatTensor(obs).unsqueeze(0)
        prob = self.model(obs).squeeze(0)
        ...

    def learn(self, obs_list, act_list, reward_list):
        ...
        prob_list = self.model(obs_list)
        ...
```

--------

## 测试算法正确性时环境的选择

|      环境      |                                           说明                                           |            特点             |
| :------------: | :--------------------------------------------------------------------------------------: | :-------------------------: |
|  CartPole-v0   |    ![cartpole](https://cdn.jsdelivr.net/gh/KibaAmor/rl-notes/env_choice/cartpole.gif)    | 模型越差，episode的时间越短 |
| MountainCar-v0 | ![mountaincar](https://cdn.jsdelivr.net/gh/KibaAmor/rl-notes/env_choice/mountaincar.gif) | 模型越差，episode的时间越长 |

实现完算法，需要测试正确性时，先用episode短的环境测试，同时**结合tensorboard，将训练过程中的reward、回合的步数等信息统计出来，有利于快速判断算法是否收敛**。
