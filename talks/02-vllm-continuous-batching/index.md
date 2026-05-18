# vLLM Continuous Batching 原理深度解析

> 基于 vLLM v1 真实源码，结合 ORCA 论文，从问题出发到代码实现全面解析

---

## 一、为什么需要 Continuous Batching

### 1.1 Decode 阶段的 GPU 利用率困境

大模型推理分为两个阶段。**Prefill** 阶段一次性处理所有 prompt token，GPU 高度并行，算力接近饱和。**Decode** 阶段则截然不同——每步只生成 1 个新 token，但需要读取所有历史 token 的 KV Cache。

这造成了严重的不对称：每步计算量极小，但读取显存的数据量很大。以 A100 为例，单请求 Decode 时 GPU 算力利用率通常低于 **5%**——95% 以上的算力都在等待显存数据传输，而不是在做计算。

解法直接：**同时跑越多请求，GPU 利用率就越高**。多个请求共享一次 KV Cache 读取，带宽成本被摊薄，计算量随 batch size 线性提升。

**核心结论**：Continuous Batching 的本质目标，就是让 Decode 阶段的 batch size 始终保持尽可能大。

### 1.2 静态批处理做不到这一点

直觉上的做法是静态批处理（Static Batching）：把若干请求打包成固定批次，一起跑，等全部完成再处理下一批。问题在于，**LLM 的输出长度完全不可预知**——"1+1=?" 只需 1 步 Decode，"写一篇文章" 需要数千步。

把长短不一的请求放在同一批次里，短请求完成后它的 GPU slot 只能空等，直到批次里最慢的请求结束，新请求才能进来。这就是**调度空泡（Scheduling Bubble）**：

<iframe src="/book/cb-static.html" width="100%" height="620" frameborder="0" style="border-radius: 12px; margin: 20px 0;"></iframe>

---

## 二、Continuous Batching 怎么设计

### 2.1 核心思路：把调度粒度降到 iteration 级别

2022 年 OSDI 的 ORCA 论文提出了关键洞察：**不要等整个批次完成，在每次 GPU forward pass 结束后都重新决定下一轮 batch 的组成**。有请求完成了，立刻从 waiting 队列取新请求填入空位。这样 GPU 的每个 slot 始终被有效请求占用。

这个思路要落地，需要解决一个前提问题：KV Cache 的内存管理。传统做法按 `max_tokens` 预分配连续内存，实际利用率只有 10%~30%，严重限制了并发数。vLLM 的 **PagedAttention** 借鉴操作系统虚拟内存分页，按固定大小的 block 按需分配，内存利用率提升到 90% 以上，这为大 batch size 打下了物质基础。

Continuous Batching 真正的效果如下——每次有请求完成，立刻补入新请求：

<iframe src="/book/cb-dynamic.html" width="100%" height="580" frameborder="0" style="border-radius: 12px; margin: 20px 0;"></iframe>

### 2.2 vLLM v0：实现了，但有结构性问题

vLLM v0 实现了基本的 Continuous Batching，但保留了显式的 Prefill/Decode 阶段区分，同一个 batch 里两种状态不能共存，且抢占时需要把 KV Cache 通过 PCIe 写出到 CPU：

<iframe src="/book/cb-v0.html" width="100%" height="560" frameborder="0" style="border-radius: 12px; margin: 20px 0;"></iframe>

### 2.3 vLLM v1：用一个计数器替代整个状态机

v1 做了根本性的重构：用一个 `num_computed_tokens` 计数器替代整个状态机。每个请求只需要知道"已计算了多少 token"和"总共需要计算多少 token"，Prefill 和 Decode 在调度器眼里没有区别。抢占也因此简化：直接释放 KV blocks，不再需要 PCIe Swap：

<iframe src="/book/cb-v1.html" width="100%" height="580" frameborder="0" style="border-radius: 12px; margin: 20px 0;"></iframe>

---

## 三、vLLM v1 如何实现

以**一次请求从进入系统到返回结果**为主线，逐步展开代码。

| 文件 | 职责 |
|------|------|
| `vllm/v1/request.py` | 请求数据结构 |
| `vllm/v1/engine/core.py` | 引擎主循环 |
| `vllm/v1/core/sched/scheduler.py` | 调度器 |
| `vllm/v1/core/kv_cache_manager.py` | KV Cache 内存管理 |

### 3.1 请求到达：创建 Request，进入 waiting 队列

```python
# engine/core.py:334
def add_request(self, request: Request):
    self.scheduler.add_request(request)
    # → self.waiting.append(request)
```

`Request` 最核心的字段是 `num_computed_tokens`，它从 0 开始，随着 Prefill 和 Decode 的推进不断增加，是调度器追踪进度的唯一依据：

```python
# vllm/v1/request.py
class Request:
    request_id: str
    prompt_token_ids: list[int]   # 输入 prompt
    _output_token_ids: list[int]  # 已生成 output

    num_computed_tokens: int      # v1 核心：统一追踪进度，初始为 0

    status: RequestStatus         # WAITING → RUNNING → FINISHED
    num_preemptions: int          # 被抢占次数
    arrival_time: float           # 到达时间（FCFS 排序依据）
    priority: int                 # 优先级（Priority 调度依据）
```

**`num_tokens` 是如何确定的？**

```python
# request.py
@property
def num_tokens(self) -> int:
    return len(self._all_token_ids)  # prompt + 已生成 output，随 Decode 增长
```

`num_tokens` 不是提前确定的，在 Decode 阶段每轮都会 +1：每次 `update_from_output()` 追加一个新 output token，`num_tokens` 随之增长，而 `num_computed_tokens` 在下一轮追上它，如此循环直到 EOS。

需要说明的是，**正常 Prefill 是一次完成的**——`num_computed_tokens` 在第一轮就从 0 直接跳到 `num_prompt_tokens`，不存在多轮差值。只有开启 **Chunked Prefill**（prompt 极长时分块处理）才会出现 Prefill 跨多轮追赶的情况。

状态机只有几种状态（`request.py:316`）：

```python
class RequestStatus(enum.IntEnum):
    WAITING    = auto()  # 在 waiting 队列
    RUNNING    = auto()  # 参与当前 batch
    PREEMPTED  = auto()  # 被抢占，KV 已释放，放回 waiting 头部
    FINISHED_STOPPED       = auto()  # 遇到 EOS
    FINISHED_LENGTH_CAPPED = auto()  # 达到 max_tokens
    FINISHED_ABORTED       = auto()  # 用户取消
```

### 3.2 主循环：`step()` 驱动三步流水

引擎在持续循环里反复调用 `step()`，每次对应一次 GPU forward pass（`engine/core.py:425`）：

```python
def step(self):
    # ① 调度：决定本轮 batch，同时在 GPU 运行前更新 num_computed_tokens
    scheduler_output = self.scheduler.schedule()

    # ② 执行：GPU forward pass
    future = self.model_executor.execute_model(scheduler_output, non_block=True)
    model_output = future.result()

    # ③ 更新：追加 output token，检查完成条件
    return self.scheduler.update_from_output(scheduler_output, model_output)
```

### 3.3 调度核心：`schedule()` 的两个阶段

`schedule()` 分两个阶段执行（`scheduler.py:310`）。

**Phase 1：优先保障 running 队列中的现有请求**

```python
# scheduler.py:329
token_budget = self.max_num_scheduled_tokens  # 默认 8192

for request in self.running:
    num_new_tokens = request.num_tokens - request.num_computed_tokens
    # Chunked Prefill：prompt 过长时分块，避免独占整个 batch
    if 0 < threshold < num_new_tokens:
        num_new_tokens = threshold
    num_new_tokens = min(num_new_tokens, token_budget)

    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)
    if new_blocks is not None:
        token_budget -= num_new_tokens
    else:
        # 内存不足 → 抢占 running 末尾的请求（见 3.4 节）
        preempted_req = self.running.pop()
        self._preempt_request(preempted_req, timestamp)
```

**Phase 2：从 waiting 补充新请求（仅当本轮无抢占）**

```python
# scheduler.py:526
# 有抢占时跳过——刚释放的内存应优先让被抢占请求恢复
if not preempted_reqs:
    while self.waiting and token_budget > 0:
        request = self.waiting.peek_request()
        # prefix cache 检测：被抢占后恢复时，命中则跳过部分重算
        new_computed_blocks, num_cached = self.kv_cache_manager.get_computed_blocks(request)
        new_blocks = self.kv_cache_manager.allocate_slots(request, ...)
        if new_blocks:
            self.running.append(request)
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = num_cached  # 从命中位置继续
```

**关键细节：`_update_after_schedule()` 在 GPU 运行前提前推进计数器**

```python
# scheduler.py:901 — schedule() 末尾，GPU 尚未运行
self._update_after_schedule(scheduler_output)

# 内部：
request.num_computed_tokens += num_scheduled_token
```

提前更新是为了让 Chunked Prefill 的下一块在下一轮调度时能立刻计算正确的断点，实现流水线式连续 Prefill。

### 3.4 内存不足时的抢占：`_preempt_request()`

```python
# scheduler.py:910
def _preempt_request(self, request: Request, timestamp: float) -> None:
    self.kv_cache_manager.free(request)   # 释放 KV blocks，显存立刻归还

    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0        # 进度清零
    request.num_preemptions += 1

    self.waiting.prepend_request(request)  # 放回 waiting 头部，下轮优先恢复
```

FCFS 模式下，`self.running.pop()` 弹出最晚加入 running 的请求；Priority 模式下，选 `priority` 数值最大（优先级最低）的请求。

### 3.5 输出处理：`update_from_output()`

```python
# scheduler.py:1248
def update_from_output(self, scheduler_output, model_runner_output):
    for req_id in num_scheduled_tokens:
        request = self.requests[req_id]
        generated_token_ids = sampled_token_ids[req_index]

        request.append_output_token_ids(generated_token_ids)  # 追加新 token

        if check_stop(request):   # EOS / max_tokens / 停止词
            self.kv_cache_manager.free(request)
            request.status = RequestStatus.FINISHED_STOPPED
            self.running.remove(request)
```

### 3.6 完整生命周期

下面的动画把以上所有步骤串在一起，展示一次请求从进入 waiting 队列到返回结果的完整过程，包括 Prefill、Decode、抢占（注意 output token 消失的瞬间）和恢复：

<iframe src="/book/cb-scheduler-loop.html" width="100%" height="720" frameborder="0" style="border-radius: 12px; margin: 20px 0;"></iframe>

---

## 总结

Continuous Batching 解决的是 Decode 阶段 GPU 利用率低的问题，核心是把调度粒度从请求级降到 iteration 级，让 batch size 始终最大化。

vLLM v1 用 `num_computed_tokens` 一个计数器统一了 Prefill 和 Decode，使调度器不需要区分阶段，所有复杂特性（Chunked Prefill、prefix cache、抢占恢复）在这个框架下自然支持。正如源码注释（`scheduler.py:311`）：

```python
# There's no "decoding phase" nor "prefill phase" in the scheduler.
# Each request just has the num_computed_tokens and num_tokens_with_spec.
# At each step, the scheduler tries to assign tokens to the requests
# so that each request's num_computed_tokens can catch up its num_tokens_with_spec.
```

---

## 参考资料

1. [vLLM GitHub](https://github.com/vllm-project/vllm)
2. [vLLM v1 设计文档](https://docs.vllm.ai/en/latest/design/v1_design.html)
3. [ORCA: A Distributed Serving System (OSDI'22)](https://www.usenix.org/system/files/osdi22-yu.pdf)
4. [PagedAttention (SOSP'23)](https://arxiv.org/abs/2309.06180)
5. [知乎：图文详解 Continuous Batch](https://zhuanlan.zhihu.com/p/1117099341)
6. [知乎：大模型推理优化：Continuous Batching](https://zhuanlan.zhihu.com/p/719610083)
