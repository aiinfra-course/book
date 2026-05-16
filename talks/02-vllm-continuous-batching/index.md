# vLLM Continuous Batching 原理深度解析

> 基于 vLLM v1 真实源码，由浅入深讲解大模型推理的核心调度技术

---

## 一、LLM 推理的两个核心阶段

大语言模型生成文本分为两个截然不同的阶段：

### 1.1 Prefill（预填充）

**目标**：处理用户输入的 Prompt，一次性计算全部 KV Cache。

| 特性 | 说明 |
|------|------|
| 计算模式 | **计算密集型**：所有 prompt tokens 并行处理 |
| 内存操作 | 写入完整的 KV Cache 到 HBM |
| 执行次数 | 仅执行 **1 次** |
| GPU 利用 | 矩阵乘高度并行，算力打满 |

### 1.2 Decode（自回归解码）

**目标**：每次生成一个新 token，循环执行直到结束。

| 特性 | 说明 |
|------|------|
| 计算模式 | **内存带宽密集型**：每步仅处理 1 个 token，但要读取全量 KV |
| 内存操作 | 读取所有历史 KV Cache + 写入新 token 的 KV |
| 执行次数 | **N 次**（等于输出 token 数） |
| GPU 利用 | 每步计算量极小，GPU 严重欠载 |

### 1.3 可视化演示

<iframe src="/book/demo.html" width="100%" height="950" frameborder="0" style="border-radius: 12px; margin: 20px 0;"></iframe>

### 1.4 两阶段对比

| 特性 | Prefill | Decode |
|------|---------|--------|
| 处理 token 数 | 全部 prompt tokens | 每次 1 个 |
| 计算类型 | 计算密集（FLOP-bound） | 内存带宽密集（memory-bound） |
| 执行次数 | 1 次 | 输出长度 N 次 |
| KV Cache 操作 | 写入 | 读取 + 追加 |
| GPU SM 利用率 | 高 | 低（通常 < 10%） |

---

## 二、传统静态批处理的问题

### 2.1 静态批处理模型

静态批处理（Static Batching）将固定数量的请求打包处理，**批次大小不变**直到所有请求完成：

```
Batch 开始时：
  [Req A: prompt=50, output=?]
  [Req B: prompt=30, output=?]
  [Req C: prompt=80, output=?]

↓ 开始执行，大小固定为 3

直到所有请求全部完成，才能接受新请求
```

### 2.2 调度空泡（Scheduling Bubble）

```
静态批处理时间线:

         t: 1  2  3  4  5  6  7  8  9  10
   Req A:   P  P  D  D  D  ✓  .  .  .  .     (P=Prefill, D=Decode)
   Req B:   P  D  D  ✓  .  .  .  .  .  .
   Req C:   P  P  P  D  D  D  D  D  D  ✓

   GPU:    [■  ■  ■  ■  ■  ■  ■  ■  ■  ■]   看起来忙碌...
   实际:   [■  ■  ■  □  □  □  ■  ■  ■  ■]   但 A/B 完成后 GPU 空等！
   (□ = GPU 在运行 C，但 A/B 的 slot 浪费了)
```

**根本问题**：
- Req B 在 t=4 就完成，但 GPU **必须等到 t=10** 才能接受新请求
- 生成长度不一致 → 短请求完成后 GPU 利用率下降
- 内存始终被所有请求占用，新请求无法插入

### 2.3 静态 vs 动态批处理动画

<iframe src="/book/cb-static-vs-dynamic.html" width="100%" height="620" frameborder="0" style="border-radius: 12px; margin: 20px 0;"></iframe>

---

## 三、Continuous Batching 的解决方案

### 3.1 核心思想：迭代级调度

> **以 Token 为调度粒度**，每个推理迭代（forward pass）结束后动态调整批次组成。

```
Continuous Batching 时间线:

t=0: [Req A: P] [Req B: P] [Req C: P]    ← 3 个请求同时 Prefill
t=1: [Req A: D] [Req B: D] [Req C: D]    ← 全部进入 Decode
t=2: [Req A: D] [Req B: D] [Req C: D]
t=3: [Req A: D] [Req B: ✓] [Req C: D]    ← B 完成！立即释放 slot
t=3: [Req A: D] [Req D: P] [Req C: D]    ← 同一 iteration 加入 Req D
t=4: [Req A: ✓] [Req D: D] [Req C: D]    ← A 完成，继续加入 Req E
...
GPU 始终满负荷运行 ✅
```

**关键差异**：不再等待整个批次完成，每次 forward pass 结束都重新调度。

### 3.2 Token Budget 机制

vLLM v1 使用 **Token Budget** 控制每轮调度的总计算量：

```python
# scheduler.py:329
token_budget = self.max_num_scheduled_tokens  # 默认 8192

# 每个请求消耗 budget
num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens
num_new_tokens = min(num_new_tokens, token_budget)

token_budget -= num_new_tokens  # 扣减预算
```

这确保每个 batch 的 token 总量受控，防止 OOM。

---

## 四、v0 vs v1 架构演进

### 4.1 v0：三队列 + 阶段分离

```
v0 架构：

     ┌──────────┐    调度     ┌──────────┐   抢占(swap)   ┌──────────┐
     │ waiting  │ ──────────► │ running  │ ─────────────► │ swapped  │
     └──────────┘             └──────────┘                └──────────┘
                                    │
                           ┌────────┴────────┐
                           │                 │
                     Prefill 阶段        Decode 阶段
                     (不能同时!)         (不能同时!)
```

**v0 的核心痛点**：

| 问题 | 影响 |
|------|------|
| Prefill/Decode 不能同时执行 | GPU 出现 "prefill bubble" |
| 抢占需要 Swap（KV 写 CPU） | 额外的 PCIe 传输开销 |
| 三队列状态机复杂 | 难以支持 Chunked Prefill / Spec Decode |
| 显式阶段判断 | 代码耦合严重 |

### 4.2 v1：两队列 + 无阶段统一调度

```
v1 架构：

     ┌──────────┐    调度     ┌──────────┐
     │ waiting  │ ◄────────── │ running  │
     └──────────┘    抢占     └──────────┘
          │           (直接放回 waiting，无 swap)
          │
     num_computed_tokens = 0 (重置)

没有显式的 Prefill / Decode 阶段！
每个请求只有:
  - num_computed_tokens  (已算了多少)
  - num_tokens_with_spec (目标多少)
```

**v1 的革命性改变**：用一个计数器替代整个状态机。

```python
# request.py:146
self.num_computed_tokens = 0  # 唯一的进度追踪字段

# scheduler.py:311-319 — 官方注释
# NOTE(woosuk): There's no "decoding phase" nor "prefill phase" in the scheduler.
# Each request just has the num_computed_tokens and num_tokens_with_spec.
# At each step, the scheduler tries to assign tokens to the requests
# so that each request's num_computed_tokens can catch up its num_tokens_with_spec.
# This is general enough to cover chunked prefills, prefix caching,
# speculative decoding, and the "jump decoding" optimization.
```

### 4.3 v0 vs v1 全面对比

| 维度 | v0 | v1 |
|------|-----|-----|
| 队列数量 | 3 个（waiting/running/swapped） | **2 个**（waiting/running） |
| 阶段区分 | 显式 Prefill/Decode 状态 | **无阶段**（统一 num_computed_tokens） |
| 抢占策略 | Swap 到 CPU / Recompute | **直接 free + prepend_waiting** |
| Prefill+Decode 混跑 | ❌ 不支持 | ✅ 天然支持 |
| Chunked Prefill | 需要额外处理 | ✅ 自然支持 |
| Speculative Decoding | 复杂扩展 | ✅ 统一处理 |
| 代码复杂度 | 高 | **显著降低** |

---

## 五、vLLM v1 工程化实现

### 5.1 核心文件索引

| 文件 | 职责 | 关键函数 |
|------|------|---------|
| [`engine/core.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/core.py) | 引擎主循环 | `step()`, `add_request()` |
| [`core/sched/scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) | 调度器核心 | `schedule()`, `_preempt_request()` |
| [`request.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/request.py) | 请求数据结构 | `Request`, `RequestStatus` |
| [`core/kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py) | KV 缓存管理 | `allocate_slots()`, `free()` |

### 5.2 Request 数据结构

```python
# request.py
class Request:
    request_id: str
    prompt_token_ids: list[int]       # 输入 prompt tokens
    _output_token_ids: list[int]      # 生成的 output tokens
    spec_token_ids: list[int]         # Speculative Decode 的 draft tokens

    # 🔑 核心字段 — 统一进度追踪
    num_computed_tokens: int          # 已完成计算的 token 数
    num_output_placeholders: int      # 异步调度用的占位符数

    # 状态枚举
    status: RequestStatus             # WAITING / RUNNING / PREEMPTED / FINISHED_*

    # 元信息
    num_preemptions: int              # 被抢占次数（用于监控）
    arrival_time: float               # 到达时间
    priority: int                     # 优先级（支持优先级调度策略）
```

**`RequestStatus` 状态枚举**（`request.py:316`）：

```python
class RequestStatus(enum.IntEnum):
    WAITING = auto()                           # 等待调度
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = auto()  # 等待语法树初始化
    WAITING_FOR_REMOTE_KVS = auto()            # P/D 分离等待远程 KV
    RUNNING = auto()                           # 正在执行
    PREEMPTED = auto()                         # 被抢占（回到 waiting）
    FINISHED_STOPPED = auto()                  # 正常结束（遇到 EOS）
    FINISHED_LENGTH_CAPPED = auto()            # 达到 max_tokens
    FINISHED_ABORTED = auto()                  # 用户主动中止
    # ...
    # NOTE: status > PREEMPTED 均视为已完成
```

### 5.3 调度器核心结构

```python
# scheduler.py
class Scheduler:
    # 两个核心队列
    waiting: RequestQueue    # 等待调度（FIFO / Priority）
    running: list[Request]   # 正在运行

    # 约束参数
    max_num_running_reqs: int         # 最大并发请求数（max_num_seqs）
    max_num_scheduled_tokens: int     # 每轮最大 token 预算（默认 8192）
    max_model_len: int                # 模型最大序列长度
```

---

## 六、调度算法深度解析

### 6.1 每轮 schedule() 完整流程

```
schedule() 调用顺序：

1. 初始化 token_budget = max_num_scheduled_tokens (8192)

2. 遍历 running 队列（FCFS 顺序）:
   for request in self.running:
       ├── 计算 num_new_tokens = num_tokens_with_spec - num_computed_tokens
       ├── num_new_tokens = min(num_new_tokens, token_budget)
       ├── allocate_slots(request, num_new_tokens)
       │   ├── 成功 → 加入 scheduled_running_reqs，token_budget -= N
       │   └── 失败（内存不足）→ 触发抢占
       └── 抢占后重试

3. 如果 token_budget > 0 且 running < max_running:
   while waiting and token_budget > 0:
       request = waiting.peek()
       ├── 检查 prefix cache 命中 → num_computed_tokens 可能直接跳过
       ├── allocate_slots(request, num_new_tokens)
       ├── 成功 → running.append(request), token_budget -= N
       └── 失败 → break（内存不足，等下轮）

4. 构建 SchedulerOutput 返回给 executor
```

### 6.2 Token Budget 控制代码（scheduler.py:366-479）

```python
def schedule(self) -> SchedulerOutput:
    token_budget = self.max_num_scheduled_tokens  # 默认 8192

    # 遍历 running 队列
    req_index = 0
    while req_index < len(self.running) and token_budget > 0:
        request = self.running[req_index]

        # 计算本轮需要处理的新 token 数（Prefill 或 Decode 均适用）
        num_new_tokens = (
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens
        )

        # Chunked Prefill: 超长 prefill 分块处理
        if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
            num_new_tokens = self.scheduler_config.long_prefill_token_threshold

        # 不超过总预算
        num_new_tokens = min(num_new_tokens, token_budget)

        # 尝试分配 KV Cache blocks
        new_blocks = self.kv_cache_manager.allocate_slots(
            request, num_new_tokens,
            num_lookahead_tokens=self.num_lookahead_tokens,
        )

        if new_blocks is not None:
            # 分配成功，加入本轮调度
            scheduled_running_reqs.append(request)
            token_budget -= num_new_tokens
            req_index += 1
        else:
            # 内存不足，抢占最低优先级请求
            preempted_req = self.running.pop()  # FCFS: 最后加入的优先被抢占
            self._preempt_request(preempted_req, scheduled_timestamp)
```

### 6.3 抢占机制（scheduler.py:910）

```python
def _preempt_request(self, request: Request, timestamp: float) -> None:
    """
    v1 简化抢占: 不再 swap to CPU，直接释放 + 重置 + 放回队列
    """
    assert request.status == RequestStatus.RUNNING

    # Step 1: 释放所有 KV Cache blocks → 归还物理内存
    self.kv_cache_manager.free(request)
    self.encoder_cache_manager.free(request)

    # Step 2: 更新状态 + 重置进度计数器
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0   # 下次重新从头计算（或利用 prefix cache 跳过）
    request.num_preemptions += 1      # 记录被抢占次数

    # Step 3: 放回 waiting 队列头部（高优先级）
    self.waiting.prepend_request(request)
    # 下轮调度时 prefix cache 可能命中，跳过已计算部分 ⚡
```

**与 v0 对比**：

| 操作 | v0 Swap 策略 | v1 简化策略 |
|------|------------|------------|
| KV Cache 处理 | 写出到 CPU 内存 | **直接释放** |
| 恢复方式 | Swap In（PCIe 回传） | **重新计算**（prefix cache 加速） |
| 额外队列 | swapped 队列 | **无** |
| 延迟影响 | PCIe 带宽瓶颈 | 依赖 prefix cache 命中率 |

### 6.4 Chunked Prefill

当 prompt 极长时（如 32K tokens），全量 Prefill 会独占整个 batch，导致已在 decode 中的请求等待过久。Chunked Prefill 将长 Prefill 拆分成多个小块：

```python
# scheduler.py:371-372
if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = self.scheduler_config.long_prefill_token_threshold
    # 下一轮继续从 num_computed_tokens + N 开始

# 效果：
# 原本: [Prefill 32K tokens] → 阻塞 decode 32 步
# 现在: [Prefill 512] [Decode A,B] [Prefill 512] [Decode A,B] ...
#        Prefill 和 Decode 交替执行，延迟更均匀
```

---

## 七、完整执行流程

### 7.1 请求生命周期

<iframe src="/book/cb-scheduler-loop.html" width="100%" height="680" frameborder="0" style="border-radius: 12px; margin: 20px 0;"></iframe>

### 7.2 代码调用链

```
用户 API 请求
    ↓
LLMEngine.generate()
    ↓
EngineCore.add_request()
    → waiting 队列
    ↓
[主循环] EngineCore.step()
    ↓
    ├─ Scheduler.schedule()          ← scheduler.py:310
    │   ├─ 遍历 running              ← scheduler.py:346
    │   ├─ allocate_slots()          ← kv_cache_manager.py
    │   ├─ 内存不足 → _preempt()     ← scheduler.py:910
    │   └─ 从 waiting 补充新请求     ← scheduler.py:526
    │
    ├─ Executor.execute_model()      ← GPU forward pass
    │   └─ PagedAttention / FlashAttention
    │
    ├─ Scheduler.update_from_output()
    │   ├─ num_computed_tokens += N  ← scheduler.py:932
    │   ├─ append_output_token_ids() ← request.py:217
    │   └─ check_stop() → FINISHED  ← sched/utils.py
    │
    └─ 返回完成的请求给用户
```

---

## 八、PagedAttention KV Cache 管理

### 8.1 为什么需要 PagedAttention

传统 KV Cache 按 max_tokens 预分配连续内存：

```
传统方式 (max_tokens=2048):
Req A: [████████████████ 2048 tokens ████████████████]  ← 实际只用 200 tokens
Req B: [████████████████ 2048 tokens ████████████████]  ← 浪费 1848 tokens
Req C: [████████████████ 2048 tokens ████████████████]  ← 内部碎片严重!

内存利用率: 10% ~ 30%  ← 极度浪费
```

PagedAttention 按需分配 block：

```
PagedAttention (block_size=16):
Req A: [Block#0][Block#3][Block#7]     ← 按需分配，物理不连续
Req B: [Block#1][Block#4]              ← 非连续内存
Req C: [Block#0][Block#2][Block#5]     ← Block#0 与 A 共享 (prefix cache!)
       ↑ 共享!

内存利用率: > 90%  ← 接近零碎片
```

### 8.2 KV Cache 块管理动画

<iframe src="/book/cb-kv-cache-blocks.html" width="100%" height="680" frameborder="0" style="border-radius: 12px; margin: 20px 0;"></iframe>

### 8.3 Block 分配流程

```python
# kv_cache_manager.py: allocate_slots()
def allocate_slots(request, num_new_tokens, num_lookahead_tokens=0):
    """
    为请求分配足够的 KV cache blocks
    返回: KVCacheBlocks（成功） 或 None（内存不足）
    """
    # 1. 检查 prefix cache 命中（跳过已缓存块）
    computed_blocks = get_computed_blocks(request)
    request.num_computed_tokens = len(computed_blocks) * block_size

    # 2. 计算需要新分配的 blocks 数量
    num_required_blocks = ceil(num_new_tokens / block_size)

    # 3. 从空闲池分配
    if len(free_blocks) < num_required_blocks:
        return None  # 触发抢占

    new_blocks = [free_blocks.pop() for _ in range(num_required_blocks)]
    return KVCacheBlocks(blocks=(computed_blocks + new_blocks,))
```

### 8.4 Prefix Cache 工作原理

```
场景: 多个请求共用系统 Prompt

Req A: [Sys Prompt: 16 tokens][用户问题 A]
Req B: [Sys Prompt: 16 tokens][用户问题 B]
Req C: [Sys Prompt: 16 tokens][用户问题 C]

第一次计算:
  Req A 的 Sys Prompt → Block#0 (hash 值: 0xABCD)
  写入 cached_block_hash_map[0xABCD] = Block#0

后续请求命中:
  Req B: hash(tokens[0:16]) == 0xABCD → 命中!
  → num_computed_tokens += 16（直接跳过，不重计算）
  → 只分配用户问题部分的 blocks

节省: 每个后续请求节省 16 tokens 的 Prefill 计算 ⚡
```

---

## 九、调度策略

### 9.1 支持的策略

```python
# core/sched/request_queue.py
class SchedulingPolicy(Enum):
    FCFS = "fcfs"          # 先来先服务（默认）
    PRIORITY = "priority"  # 按 request.priority 排序
```

**PRIORITY 策略的抢占选择**（`scheduler.py:437-441`）：

```python
# 优先级调度：抢占优先级最低的请求
preempted_req = max(
    self.running,
    key=lambda r: (r.priority, r.arrival_time),
)
```

### 9.2 调度约束参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `max_num_seqs` | 256 | 最大并发请求数 |
| `max_num_scheduled_tokens` | 8192 | 每轮 token 预算 |
| `max_num_batched_tokens` | 8192 | 每批次最大 token 数 |
| `long_prefill_token_threshold` | 0（关闭） | Chunked Prefill 阈值 |

---

## 十、总结

### 10.1 Continuous Batching 本质

> **把 GPU 每次 forward pass 的 token 预算填满**，以最细粒度（iteration 级）管理请求生命周期，让 GPU 永不空等。

### 10.2 vLLM v1 核心设计哲学

| 原则 | v0 做法 | v1 做法 |
|------|---------|---------|
| 进度追踪 | 状态机（Prefill/Decode 状态） | **计数器**（`num_computed_tokens`） |
| 阶段管理 | 显式阶段切换 | **无阶段**（统一算法覆盖所有场景） |
| 内存管理 | 预分配连续内存 | **PagedAttention**（按需分配，block 级） |
| 抢占策略 | Swap to CPU | **Free + Recompute**（prefix cache 加速） |
| 队列设计 | 3 个队列 | **2 个队列** |

### 10.3 一个设计统一所有优化

```python
# scheduler.py:311 — 这段注释道破 v1 设计核心
#
# There's no "decoding phase" nor "prefill phase" in the scheduler.
# Each request just has num_computed_tokens and num_tokens_with_spec.
#
# This is general enough to cover:
#   ✅ Chunked Prefills      (长 prompt 分块)
#   ✅ Prefix Caching        (跨请求 KV 共享)
#   ✅ Speculative Decoding  (draft token 验证)
#   ✅ Multi-modal           (图文混合)
```

---

## 参考资料

1. [vLLM GitHub](https://github.com/vllm-project/vllm)
2. [vLLM v1 设计文档](https://docs.vllm.ai/en/latest/design/v1_design.html)
3. [ORCA: A Distributed Serving System (OSDI'22)](https://www.usenix.org/system/files/osdi22-yu.pdf) — Continuous Batching 奠基论文
4. [vLLM: Easy, Fast, and Cheap LLM Serving (SOSP'23)](https://arxiv.org/abs/2309.06180) — PagedAttention 论文
5. [知乎：图文详解 Continuous Batch](https://zhuanlan.zhihu.com/p/1117099341)
6. [知乎：大模型推理优化：Continuous Batching](https://zhuanlan.zhihu.com/p/719610083)
