# Continuous Batching 设计深度解析

> 基于 vLLM v1 真实源码，由浅入深讲解大模型推理的核心优化技术

## 目录
1. [问题引入：为什么需要批处理？](#问题引入为什么需要批处理)
2. [传统批处理的瓶颈](#传统批处理的瓶颈)
3. [Continuous Batching 核心思想](#continuous-batching-核心思想)
4. [v0 vs v1 架构演进](#v0-vs-v1-架构演进)
5. [vLLM v1 工程化实现](#vllm-v1-工程化实现)
6. [核心代码深度分析](#核心代码深度分析)
7. [完整执行流程](#完整执行流程)
8. [总结](#总结)

---

## 问题引入：为什么需要批处理？

### 单请求处理 (No Batching)

```
GPU 时间线：
Req1: |-----------|
Req2:             |-----------|
Req3:                          |-----------|
GPU:  [==========][==========][==========]
```

- 每次只处理一个请求
- 简单灵活，但 GPU 利用率极低（大部分时间在等待）

### 静态批处理 (Static Batching)

```
Batch:  Req1 Req2 Req3
       |-----------|-----------|-----------|
GPU:   [====================]

问题：Req1 完成后，GPU 只能等待其他请求！
```

- 打包多个请求一起处理
- 一旦打包，大小不再变化
- 不同请求输出长度差异巨大 → GPU 空等

---

## 传统批处理的瓶颈

### 1. 变长输入输出

大模型推理中：
- **输入长度**：Prompt 差异大（10 tokens vs 2000 tokens）
- **输出长度**：更不一致（10 tokens vs 1000 tokens）

### 2. 空等问题 (Scheduling Bubble)

```
静态批处理时间线:
Req1:  P P D D D D ✓
Req2:  P D D ✓
Req3:  P P P D D D D D D ✓
       ──────────────
时间:  1 2 3 4 5 6 7 8 9 10

问题：Req2 在时间3 就完成，但 GPU 必须等到时间10！
```

---

## Continuous Batching 核心思想

### 迭代级调度 (Iteration-Level Scheduling)

> 在每个推理迭代中，动态管理批处理大小，有请求完成立即补充新请求。

```
Continuous Batching 时间线:
Time1: Req1(P) Req2(P) Req3(P)   -- 3个请求一起 Prefill
Time2: Req1(D) Req2(D) Req3(D)   -- Decode 阶段
Time3: Req1(D) Req2(✓) Req3(D)   -- Req2 完成！GPU 不等待
Time4: Req1(D) Req4(P) Req3(D)   -- 立即加入 Req4
Time5: Req1(✓) Req4(D) Req3(D)   -- Req1 完成，继续加入
...
```

**核心优势**：GPU 始终保持忙碌，最大化利用率！

---

## v0 vs v1 架构演进

### vLLM 版本时间线

```
vLLM 发展历程：
v0.2.x ── v0.3.x ── v0.4.x ── ... ── v1.x
  ↓        ↓         ↓               ↓
 基础    Paged    持续优化         重大重构
 (2023)  Attention (2023)          (2024)
```

::: tip 说明
知乎上的大量文章描述的是 **v0 版本**的架构。v1 版本已经进行了**完全重构**，设计理念完全不同。
:::

### v0 架构：三队列 + 阶段分离

```
┌─────────┐   ┌─────────┐   ┌─────────┐
│ waiting │ → │ running │ → │ swapped │ (抢占后)
└─────────┘   └─────────┘   └─────────┘
                  ↓
        ┌────────┴────────┐
        ↓                 ↓
    Prefill状态      Decode状态
        ↓                 ↓
    不能同时运行     不能同时运行
```

**v0 核心问题**：
| 问题 | 影响 |
|------|------|
| 阶段分离 | Prefill 和 Decode **不能同时执行** |
| 抢占复杂 | Swap/Recompute 两种策略难以平衡 |
| 状态机复杂 | 难以扩展 Chunked Prefill、Spec Decode |
| 队列多 | waiting/running/swapped 三队列难维护 |

### v1 架构：两队列 + 无阶段设计

```
┌─────────┐   ┌─────────┐
│ waiting │ ← │ running │ ← 抢占后直接回来
└─────────┘   └─────────┘
                  ↓
         ┌────────┴────────┐
         ↓                 ↓
     Prefill        Decode
         ↓                 ↓
     num_computed_tokens    ↓
         ↓                 ↓
         ← 可以同时进行！ →
```

**v1 革命性改进**：

| 维度 | v0 | v1 |
|------|-----|-----|
| 队列数量 | 3个 | **2个** |
| 阶段区分 | 显式（Prefill/Decode） | **无（统一计数器）** |
| 抢占机制 | 复杂（Swap/Recompute） | **简化（放回waiting）** |
| Prefill+Decode | 不能同时 | **可以同时** |
| 代码复杂度 | 高 | **低** |
| 扩展性 | 困难 | **强** |

---

## vLLM v1 工程化实现

### 两大核心队列

```python
self.waiting    # 等待调度的请求队列（RequestQueue）
self.running    # 正在推理的请求列表（list）
```

**注意**：v1 版本**没有 swapped 队列**！

### Request 核心字段

```python
class Request:
    req_id: str
    prompt_token_ids: list[int]       # 输入 token
    output_token_ids: list[int]        # 输出 token
    spec_token_ids: list[int]         # 投机解码的 draft tokens

    num_computed_tokens: int          # 🔑 核心！已计算的 token 数
    num_tokens_with_spec: int         # 总 token 数

    status: RequestStatus             # WAITING / RUNNING / PREEMPTED / FINISHED_*
```

**关键点**：v1 用 `num_computed_tokens` 统一追踪，不再区分 Prefill/Decode！

---

## 核心代码深度分析

### 1. 调度器核心设计 (scheduler.py:310-320)

```python
def schedule(self) -> SchedulerOutput:
    # NOTE(woosuk) on the scheduling algorithm:
    # There's no "decoding phase" nor "prefill phase" in the scheduler.
    # Each request just has the num_computed_tokens and num_tokens_with_spec.
    #
    # num_tokens_with_spec = len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids)
    #
    # At each step, the scheduler tries to assign tokens to the requests
    # so that each request's num_computed_tokens can catch up its num_tokens_with_spec.
```

**设计精髓**：一个通用算法覆盖所有场景！

```python
- Chunked Prefills       ✅
- Prefix Caching        ✅
- Speculative Decoding  ✅
- Multi-modal           ✅
```

### 2. Token-Level Budget 调度

**代码位置**：[scheduler.py:345-470](file:///Users/game-netease/Desktop/project/github/vllm/vllm/v1/core/sched/scheduler.py#L345-L470)

```python
token_budget = self.max_num_scheduled_tokens  # 默认 8192 tokens

# 遍历 running 请求
req_index = 0
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]

    # 计算本轮需要处理多少新 token
    num_new_tokens = (
        request.num_tokens_with_spec
        + request.num_output_placeholders
        - request.num_computed_tokens
    )

    # 确保不超过本轮预算
    num_new_tokens = min(num_new_tokens, token_budget)

    # 尝试分配 KV Cache
    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)

    if new_blocks is not None:
        # 分配成功，加入本轮调度
        scheduled_running_reqs.append(request)
        token_budget -= num_new_tokens
        req_index += 1
    else:
        # 内存不足，需要抢占
        self._preempt_request(preempted_req, scheduled_timestamp)
```

### 3. 抢占机制简化

**代码位置**：[scheduler.py:910-930](file:///Users/game-netease/Desktop/project/github/vllm/vllm/v1/core/sched/scheduler.py#L910-L930)

```python
def _preempt_request(self, request: Request, timestamp: float) -> None:
    # 1. 释放 KV Cache
    self.kv_cache_manager.free(request)

    # 2. 更新状态
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0  # 重置

    # 3. 直接放回 waiting 队列头部（高优先级！）
    self.waiting.prepend_request(request)
```

**v0 vs v1 抢占对比**：

| v0 | v1 |
|-----|-----|
| Swap 策略：KV Cache 换出到 CPU | 释放 KV Cache |
| Swapped 队列 | 直接放回 waiting |
| 需要 Swap In 才能恢复 | prepend_request 高优先级 |

### 4. Waiting 队列调度

```python
# 检查是否达到最大并发数
if len(self.running) == self.max_num_running_reqs:
    break

# 取出队首请求
request = self.waiting.peek_request()

# 检查 Prefix Cache 是否命中
if request.num_computed_tokens == 0:
    computed = self.kv_cache_manager.get_computed_blocks(request)

# 计算实际需要新计算的 token
num_new_tokens = request.num_tokens - num_computed_tokens

# 尝试分配并加入 running
if allocate_slots(request, num_new_tokens):
    self.running.append(request)
```

---

## 完整执行流程

```
用户请求
   ↓
add_request()
   ↓
进入 waiting 队列
   ↓
[Loop] 引擎 step() 循环
   ↓
   ├─ Scheduler.schedule()
   │   ├─ 遍历 running，分配 token 预算
   │   ├─ 如果还有 budget，调度 waiting
   │   └─ 如果内存不够 → 抢占
   ↓
   ├─ ModelExecutor.execute_model()
   │   └─ PagedAttention / FlashAttention 计算
   ↓
   ├─ 更新 Request 状态
   │   └─ num_computed_tokens += 1
   ↓
   └─ 检查是否 finished
       ├─ 是 → 释放 KV Cache → FINISHED
       └─ 否 → 继续在 running 队列中
```

---

## 总结

### v1 版本核心优势

| 对比项 | v0（旧版） | v1（新版） |
|--------|------------|------------|
| 队列数量 | 3个（复杂） | **2个（简洁）** |
| 抢占机制 | swap/recompute | **统一放回waiting** |
| 代码复杂度 | 高 | **低** |
| 可维护性 | 较难 | **易** |

### 设计精髓

1. **状态 → 计数器**：用 `num_computed_tokens` 替代状态机
2. **有阶段 → 无阶段**：统一 token 调度，Prefill/Decode 可同时
3. **三队列 → 两队列**：移除 swapped 队列
4. **抢占简化**：直接 prepend_request 高优先级

### 关键文件索引

| 文件 | 作用 |
|------|------|
| [scheduler.py](file:///Users/game-netease/Desktop/project/github/vllm/vllm/v1/core/sched/scheduler.py) | 调度器核心 |
| [kv_cache_manager.py](file:///Users/game-netease/Desktop/project/github/vllm/vllm/v1/core/kv_cache_manager.py) | KV Cache 管理 |
| [core.py](file:///Users/game-netease/Desktop/project/github/vllm/vllm/v1/engine/core.py) | LLM 引擎主循环 |
| [request.py](file:///Users/game-netease/Desktop/project/github/vllm/vllm/v1/request.py) | Request 数据结构 |

### 与其他框架对比

| 框架 | Continuous Batching 实现 |
|------|--------------------------|
| vLLM v1 | Prefill/Decode 统一，只有 waiting/running 两队列 |
| DeepSpeed-MII | Dynamic SplitFuse，动态分割融合 |
| LMDeploy | Persistent Batch，预分配 batch slots |
| TensorRT-LLM | In-Flight Batch Manager |

---

## 参考资料

1. [ORCA: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/system/files/osdi22-yu.pdf)
2. [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
3. [vLLM GitHub](https://github.com/vllm-project/vllm)
4. [vLLM v1 设计文档](https://docs.vllm.ai/en/latest/design/v1_design.html)
5. [知乎：图文详解 Continuous Batch](https://zhuanlan.zhihu.com/p/1117099341)
6. [知乎：大模型推理优化：Continuous Batching](https://zhuanlan.zhihu.com/p/719610083)
