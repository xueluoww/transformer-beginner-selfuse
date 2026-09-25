# Transformer V0.8 — 一个 Python / PyTorch 初学者手写的可训练、可推理 Transformer

> **先声明：我是初学者。**
>
> 我开始写这个项目时，学习 Python 语法还不到一周，PyTorch 也刚学到入门阶段。这个仓库不是“标准实现”“高性能实现”或“生产级实现”，里面有不少硬编码、重复逻辑、笨拙的类设计、手工拆分、多层手写堆叠，以及很多资深开发者看了可能会头疼的地方。
>
> 这个项目最重要的目的不是展示“我写得多专业”，而是记录我如何从一个固定的 Tensor 开始，一点一点理解并搭出一个能够训练、保存、加载和自回归翻译的 Transformer。

---

## 1. V0.8 到底是什么？

V0.8 是目前第一个真正形成完整闭环的版本。

它现在已经可以完成：

```text
文本数据集
    ↓
Dataset / DataLoader
    ↓
建立或读取中英文词表
    ↓
字符级 Token 化
    ↓
Batch Padding
    ↓
Padding Mask
    ↓
Encoder × 5
    ↓
Decoder × 5
    ↓
FC 输出 logits
    ↓
CrossEntropyLoss
    ↓
Backward
    ↓
Gradient Clipping
    ↓
AdamW 更新参数
    ↓
保存 state_dict
    ↓
重新加载模型
    ↓
输入新的中文文本
    ↓
Decoder 从 <BOS> 开始自回归生成
    ↓
输出英文
```

也就是说，它已经不再只是“搭出了一个 Transformer 结构”，而是第一次能够从数据一直走到训练，再从保存好的参数走到实际翻译。

---

## 2. 当前模型结构

这并不是论文原版 Transformer 的严格复现，而是我为了理解结构而一步一步搭出来的版本。

当前主要参数：

- `d_model = 512`
- Multi-Head Attention：**4 heads**
- Encoder：**5 层**
- Decoder：**5 层**
- 字符级 Tokenizer
- Encoder / Decoder 使用独立词表
- Encoder / Decoder 使用独立 Embedding
- `PAD = 0`
- `BOS = 1`
- `EOS = 2`
- `UNK = 3`

### Encoder 每层

```text
Multi-Head Self-Attention
        ↓
Residual + LayerNorm
        ↓
manyFFN
        ↓
Residual + LayerNorm
```

### Decoder 每层

```text
Masked Multi-Head Self-Attention
        ↓
Residual + LayerNorm
        ↓
Cross Multi-Head Attention
    Q：来自 Decoder
    K/V：来自 Encoder
        ↓
Residual + LayerNorm
        ↓
manyFFN
        ↓
Residual + LayerNorm
```

我的 `manyFFN` 也不是标准 Transformer 的 FFN，而是一个为了自己理解和实验写出来的自定义结构：多个并行线性层分别计算后再拼接，最后映射回 512 维。

---

## 3. V0.8 这一版我主要做了什么？

### 3.1 Dataset / DataLoader

我自己写了一个 `Dataset`，从 `text.txt` 中逐行读取中英文句对。

数据格式类似：

```text
我爱你    i love you
今天天气很好    the weather is nice today
中秋节快乐    happy mid-autumn festival
```

程序会检查每一行是否使用 Tab 分隔，并把中英文分别保存。

训练时再交给 `DataLoader` 组成 batch。

当前训练脚本使用：

```text
batch_size = 8
shuffle = True
```

---

### 3.2 独立中英文词表

早期版本中，我使用过共享词表。

V0.8 现在已经改成：

```text
Chinese vocab
English vocab
```

分别保存和读取。

词表会持久化到文件中，下一次运行时直接加载，从而保证：

```text
训练时 token ID
=
推理时 token ID
```

否则即使字符完全一样，只要重新建立词表时 ID 顺序变化，已经训练好的 Embedding 和 FC 参数就会失去原来的含义。

---

### 3.3 Batch Tokenization + Padding

这是 V0.8 非常重要的一步。

以前我的代码一次只处理一个句子，现在会：

1. 找到当前 batch 中最长的 Encoder 序列；
2. 找到当前 batch 中最长的 Decoder 序列；
3. 用 `<PAD>` 补齐；
4. 构造 Tensor。

Decoder 训练输入采用：

```text
[BOS] + target
```

训练答案采用：

```text
target + [EOS]
```

例如：

```text
target:
i love you

decoder input:
[BOS] i love you

answer:
i love you [EOS]
```

---

### 3.4 Padding Mask

加入 Padding 之后，Attention 就不能再把 `<PAD>` 当成正常 token 参与注意力计算。

因此 V0.8 加入了：

- Encoder Padding Mask
- Decoder Padding Mask
- Cross-Attention 中对 Encoder Padding 的屏蔽

Padding Mask 会一路传播到 Attention score，在 softmax 之前把需要屏蔽的位置填成 `-inf`。

> 这一部分是整个项目里我最明确“直接大量采用 ChatGPT 代码”的地方。
>
> Padding Mask 的传播涉及多层函数参数传递，我理解它为什么存在、遮哪里、为什么 Cross-Attention 要使用 Encoder 的 Padding Mask，但具体传播代码我没有坚持逐行自己重新发明，而是直接采用并修改了 AI 给出的实现。

---

### 3.5 动态 Batch

早期版本里很多 reshape 都直接写死：

```python
batch = 1
```

V0.8 改成从 Tensor 自己读取：

```text
k.shape[0]
q.shape[0]
v.shape[0]
```

序列长度也不再依赖全局变量，而是直接从：

```text
k.shape
q.shape
v.shape
```

中获取。

这让 Self-Attention 和 Cross-Attention 能够根据输入 Tensor 自己判断当前 batch 和序列长度。

---

### 3.6 Position Encoding

位置编码依然是我自己按公式手写的 sinusoidal positional encoding。

目前实现并不高效，仍然使用 Python 循环逐元素构造位置矩阵。

它很慢，也绝对不是推荐的工程写法。

但我暂时保留它，因为这是我最早理解 Transformer 位置编码时写出来的实现，也是这个项目从 V0.1 一路留下来的痕迹。

---

### 3.7 完整模型封装

现在 Embedding、Transformer 主体和最终 FC 已经被封装进一个总模型：

```text
ReallyTrainModel
├── Encoder Embedding
├── Decoder Embedding
├── Transformer
│   ├── Encoder × 5
│   └── Decoder × 5
└── Output FC
```

一次：

```python
model(encoder_token, decoder_token)
```

会完成：

```text
Embedding
→ Position Encoding
→ Transformer
→ FC
→ logits
```

而自回归推理逻辑只负责反复调用模型。

---

## 4. 训练流程

V0.8 已经具有真正的训练闭环。

当前使用：

```python
CrossEntropyLoss(ignore_index=0)
AdamW
gradient clipping
```

基本流程：

```text
optimizer.zero_grad()
        ↓
model(...)
        ↓
logits reshape
        ↓
CrossEntropyLoss
        ↓
loss.backward()
        ↓
clip_grad_norm_
        ↓
optimizer.step()
```

训练完成后保存：

```text
model.state_dict()
```

目前默认训练 30 个 epoch。

一次实际训练中，后期训练集平均 loss 大约下降到了：

```text
0.06 左右
```

**注意：这是训练集 loss，不是验证集指标。**

当前数据集规模很小，也没有完整的 train / validation / test 划分，因此这个数字不能用来说明模型具有真正的泛化能力。

---

## 5. 推理 / 翻译模式

现在程序启动后可以选择：

```text
train
```

进入训练模式；

或者：

```text
translate
```

进入翻译模式。

翻译模式会：

```text
读取词表
↓
建立模型
↓
加载保存好的 state_dict
↓
model.eval()
↓
输入中文
↓
Encoder Tokenize
↓
Decoder 从 [BOS] 开始
↓
预测下一个 token
↓
拼回 Decoder
↓
继续预测
↓
直到 EOS 或达到最大长度
```

---

## 6. 当前实际输出

在目前这份只有几百条中英句对的小数据集上，模型已经能够输出一些比较有意思的结果。

例如：

```text
输入：
我爱你

输出：
i love you
```

```text
输入：
我非常爱你

输出：
i love you very much
```

```text
输入：
今晚的月色真美

输出：
the moonlight is beautiful tonight
```

```text
输入：
中秋节快乐

输出：
happy mid-autumn festival
```

也出现过一定程度的组合能力：

```text
输入：
愿我们年年相聚

输出：
may we reunite every year
```

但面对明显超出训练分布、更加文学化或复杂的句子时，模型也会生成混乱结果。

例如类似：

```text
但愿人长久，千里共婵娟
```

这样的输入，目前仍然可能出现拼写错误、词语重复、句子结构崩坏等现象。

这很正常。

目前它只是一个小数据集上训练出来的字符级 Transformer 学习项目，不应该把这些结果理解成真正成熟的机器翻译能力。

---

## 7. 我知道这份代码还有很多问题

这是我非常想提前声明的一点。

**请不要把这个仓库当成 Transformer 的标准实现参考。**

我目前能意识到的问题就包括：

- 4 个 Attention Head 仍然大量手动拆分；
- Encoder / Decoder 五层仍然有明显的硬编码；
- 很多类名、变量名和拼写并不规范；
- Position Encoding 的实现非常低效；
- 有很多重复代码；
- 部分模块职责还能继续重新设计；
- `manyFFN` 并不是标准 Transformer FFN；
- 没有 KV Cache；
- 自回归推理会重复计算大量内容；
- 没有 Beam Search；
- 没有更成熟的 tokenizer；
- 没有验证集与正规评估指标；
- 没有 BLEU 等机器翻译评估；
- 没有针对 GPU / mixed precision 做优化；
- 数据集非常小；
- 模型很容易记忆训练数据；
- 很多地方只是“能跑并且我能理解”，并不是“优雅”。

这些都是真的。

但 V0.8 暂时不会继续为了“看起来专业”而大规模重构。

这一版我想先保留下来。

因为它记录的是我第一次真正把一个 Transformer 从零散模块一路搭到训练和推理闭环的过程。

---

## 8. 从 V0.1 到 V0.8

V0.1 只有大约 3 KB。

当时甚至没有真正的数据集，也没有真正的 token 输入。

整个主流程基本只是：

```python
x = torch.ones([1, 1, 34, 512])
```

然后：

```text
手动 Position Encoding
→ Q/K/V
→ Attention
→ Residual + LayerNorm
→ FFN
→ Residual + LayerNorm
```

最后：

```python
print(output)
```

后面的版本一点一点加入了：

```text
单头 Attention
→ 多头 Attention
→ Encoder
→ Decoder
→ Cross-Attention
→ 多层堆叠
→ Tokenizer
→ Vocabulary
→ BOS / EOS
→ 自回归生成
→ 模型封装
→ Dataset
→ DataLoader
→ Batch
→ Padding
→ Padding Mask
→ Loss
→ Backward
→ Optimizer
→ Gradient Clipping
→ Save / Load
→ Train / Translate
```

我自己也没有想到一开始那个只有一个 `torch.ones()` 的文件最后会长成这样。

---

## 9. 关于 ChatGPT

这个项目必须明确写上 ChatGPT 的贡献。

如果没有 ChatGPT，这个项目大概率不会以现在这个速度走到 V0.8。

ChatGPT 在整个过程中承担了很多角色：

- 给我解释 Python / PyTorch 基础；
- 解释 `nn.Module`、`forward`、`self`；
- 解释 Tensor shape；
- 帮我理解矩阵乘法与 transpose；
- 解释 Q / K / V；
- 解释 Multi-Head Attention；
- 解释 Cross-Attention；
- 解释 Causal Mask；
- 解释 Teacher Forcing；
- 解释 BOS / EOS / PAD；
- 帮我检查 Attention 各维度；
- 在代码报错时帮助定位 shape 问题；
- 帮我区分训练逻辑和推理逻辑；
- 帮我设计模型封装边界；
- 帮我检查自回归生成流程；
- 给出 Padding Mask 的传播实现；
- 在很多我卡住的地方充当一个可以随时追问的老师和 debugger。

尤其是 **Padding Mask 的传播部分，我基本直接使用了 ChatGPT 提供的代码结构**，这是我不想模糊掉的地方。

但这个项目也不是“一句话让 AI 生成一个 Transformer”。

除了上面明确提到的 Padding Mask 传播部分，大多数代码都是我自己一行一行敲出来、运行、报错、理解、修改，再继续往下搭的。

我并不排斥 AI 辅助。

相反，这个项目本身就是我第一次真正体验：

> **人在理解和做决定，AI 负责解释、讨论、排错和提供工具，最后一起把一个复杂东西做出来。**

所以如果要给这个项目的开发方式下一个定义，我更愿意叫它：

```text
AI-assisted learning / AI-assisted development
```

而不是假装整个项目是在完全没有帮助的情况下完成的。

感谢 ChatGPT 陪我从 V0.1 一路走到了 V0.8。

---

## 10. 为什么保留这个项目？

我不知道以后会不会继续做 NLP，也不知道这个 Transformer 最后会不会真的“有用”。

但这个项目对我来说已经有一点意义了。

因为我第一次发现：

> 写代码并不一定是一件无趣的事情。

很多时候它更像是在和 Tensor、shape、模型结构以及自己脑子里的理解斗智斗勇。

看到一个东西从：

```text
torch.ones(...)
```

一路变成：

```text
请输入要翻译的文本：
我爱你

i love you
```

这种感觉挺难忘的。

所以 V0.8 先停在这里。

优化、KV Cache、更标准的实现、更大的数据集、更好的训练方式……

以后再说。

---

## 11. 最后

如果你是一个有经验的 PyTorch / Transformer 开发者，看到这里某些写法觉得非常离谱：

**你大概率是对的。**

欢迎指出问题。

只是也请记得：

> 这是一个刚学 Python 没多久的初学者，为了真正理解 Transformer 而留下的一份学习记录。

它不完美。

但它真的跑起来了。
