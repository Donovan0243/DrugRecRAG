[cite_start]这是一篇关于**使用图辅助提示（GAP）进行对话式药物推荐**的研究论文 [cite: 2]。

[cite_start]简而言之，这篇论文解决了一个核心问题：大型语言模型 (LLM) 在作为医疗对话系统时，虽然知识丰富，但常常会**忽略对话中的关键细节**（例如患者怀孕、有其他疾病）[cite: 11, 54, 55][cite_start]，或者**缺乏特定的领域知识**（例如药物禁忌症）[cite: 12, 56][cite_start]，从而导致推荐的药物不准确甚至危险 [cite: 55, 60]。

[cite_start]该论文提出的 **GAP (Graph-Assisted Prompts) 框架** [cite: 13] 就是为了解决这个问题。

---

### 论文的核心内容 (GAP 框架如何工作)

[cite_start]GAP 框架的核心思想是**“先构建图谱，再辅助提示”**，它通过一个RAG（检索增强生成）[cite: 73] 的方式，将结构化的图谱知识注入到 LLM 的推理过程中。

1.  **构建患者中心图 (Patient-centric Graph)**：
    * [cite_start]首先，框架会从对话历史中**提取关键的医疗概念**（如疾病、症状、药物）及其“状态”（如患者自述阳性、持续时间）[cite: 14, 68, 122]。
    * [cite_start]然后，它会构建一个以患者为中心的图（$\mathcal{G}_{p}$），这个图能明确地记录和维护患者在对话中提到的所有医疗信息 [cite: 14, 69, 173]。

2.  **融合外部知识图谱 (KGs)**：
    * [cite_start]框架会将这个“患者中心图”与**外部的大型医疗知识图谱**（例如 CMeKG、Disease-KB [cite: 291][cite_start]）进行链接 [cite: 15, 71]。
    * 这样一来，系统不仅知道患者“感冒了”，还能立刻从外部KG中知道“感冒”的推荐疗法、禁忌症等。

3.  **生成图辅助提示 (Graph-Assisted Prompts)**：
    * [cite_start]这是最关键的一步。GAP 会利用这个融合后的图谱，生成两种特殊的提示（Prompts）来“指导”LLM [cite: 179, 182]：
        * [cite_start]**邻域提示 (Neighborhood Prompts)**：从知识图谱中检索与当前对话最相关的事实（例如，“*ChangYanNing* 和 *Montmorillonite* 是治疗腹泻的药物”）[cite: 220, 321]。
        * [cite_start]**基于路径的提示 (Path-based Prompts)**：通过预先定义的医疗逻辑“模式”（Schema，见论文图5 [cite: 498][cite_start]），在图上查询患者特定的知识路径。例如，一个路径可能会发现（*患者* -> *怀孕*）与（*药物* -> *Loratadine* -> *禁忌* -> *怀孕*）之间的冲突 [cite: 222, 233, 329]。

4.  **最终推理**：
    * [cite_start]最后，LLM 会接收到**所有信息**：原始对话历史 + 患者中心图（文本化后）+ 邻域提示 + 路径提示 [cite: 244, 246]。
    * [cite_start]基于这些丰富的、经过图谱验证和检索的上下文，LLM 会生成更安全、更准确的药物推荐 [cite: 73]。

---

### 如何复现这项工作

[cite_start]要复现这篇论文 (Algorithm 1 [cite: 185])，你需要按照其方法论（Methodology）和实验设置（Experiment Setup）来搭建整个流程。

#### 1. 准备核心资产

1.  **基础 LLM**：
    * [cite_start]论文中主要使用了 **DeepSeek-V3** 作为信息提取、提示生成和最终响应的 LLM [cite: 277, 293]。
    * [cite_start]他们也提到了使用 **ChatGPT-3.5-Turbo (version 1106)** 进行实验 [cite: 507]。

2.  **知识图谱 (KGs)**：
    * [cite_start]你需要一个或多个医疗知识图谱。论文中使用了 **CMeKG (中文医疗知识图谱)** 和 **Disease-KB** [cite: 291]。

3.  **数据集**：
    * [cite_start]**DialMed** [cite: 251]：这是用于对话式药物推荐的主要数据集，你需要用它来训练（如果是有监督方法）或评估（LLM-based 方法）。
    * [cite_start]**LLM as Patients** [cite: 253]：这是用于评估“动态诊断访谈”的辅助数据集。

#### 2. 搭建 GAP 框架 (Section 3)

[cite_start]这是一个分步骤的流程，对应论文中的算法1 [cite: 185]：

[cite_start]**步骤 1：医疗信息提取 (Medical Information Extraction)** [cite: 122]

* **目标**：从对话中提取实体和状态。
* **实现**：
    1.  [cite_start]**提取概念 (NER)**：使用 LLM（如 DeepSeek-V3）和特定的NER提示（见论文表4 [cite: 514][cite_start]）来识别对话中的医疗概念 $\mathcal{C}$（疾病、症状等）[cite: 126, 198]。
    2.  [cite_start]**判断状态 (Slot-filling)**：对于每个概念 $c$，使用 LLM 和状态判断提示（见论文表5 [cite: 519][cite_start]），在上下文窗口（$k=1$，即前后各1轮对话 [cite: 294][cite_start]）中判断其状态（例如：`patient claims positive`）[cite: 129, 131, 201]。

[cite_start]**步骤 2：患者中心图构建 (Graph Construction)** [cite: 171]

* **目标**：构建患者图 $\mathcal{G}_{p}$ 并链接 KGs。
* **实现**：
    1.  [cite_start]**初始化图**：创建一个“患者”节点 [cite: 173]。
    2.  **添加节点和边**：将提取到的概念 $c$ 和状态 $(c, s, v)$ 添加到图中（例如：*Patient* -> *has_symptom* -> *cold*; [cite_start]*cold* -> *state* -> *positive*）[cite: 173]。
    3.  [cite_start]**实体链接**：将图中的概念（如“感冒”）**归一化**（通过编辑距离、同义词列表等方法 [cite: 175]）到你的 KG（CMeKG）中的标准实体上。
    4.  [cite_start]**邻域图**：通过这个链接，$\mathcal{G}_{p}$ 自然地获得了来自外部 KG 的邻域知识 $\mathcal{NG_{p}}$ [cite: 176]。

[cite_start]**步骤 3：提示生成 (Prompt Generation)** [cite: 179]

* **目标**：生成 NP 和 PP。
* **实现**：
    1.  **生成邻域提示 (NP)**：
        * [cite_start]使用 LLM 判断当前对话**需要哪种类型的知识**（例如，需要“治疗方案”）[cite: 220]。
        * [cite_start]从 $\mathcal{NG_{p}}$ 中检索具有相应关系（relation）的 Top-k（论文设k=3 [cite: 296][cite_start]）个事实，作为 NP [cite: 220]。
    2.  **生成基于路径的提示 (PP)**：
        * [cite_start]**定义模式 (Schema)**：你需要像论文图5 [cite: 498] [cite_start]那样，预先定义一些关键的医疗推理路径（例如：`pregnant` -> `positive` -> `symptom` -> `treatment` -> `medication`）[cite: 224]。
        * [cite_start]**路径匹配**：在你的图（$\mathcal{G}_{p}$ + $\mathcal{NG_{p}}$）中匹配这些模式，得到路径查询 $\mathcal{PS}$ [cite: 225]。
        * [cite_start]**多源检索**：使用 $\mathcal{PS}$ 从三个来源获取信息 [cite: 229]：
            * [cite_start]**KG 验证**：检查图谱中的属性（如药物的“禁忌症”）是否与患者状态（如“怀孕”）冲突 [cite: 232-234]。
            * [cite_start]**LLM 推理**：将路径作为上下文，让 LLM 进行中间推理（见论文表6 [cite: 521][cite_start]）[cite: 236]。
            * [cite_start]**互联网访问**：将路径查询改写为自然语言，搜索互联网，获取最新信息 [cite: 240-242]。
        * [cite_start]将这三部分整合为 Top-k（论文设k=3 [cite: 296]）个 PP。

[cite_start]**步骤 4：推理和响应生成 (Inference)** [cite: 243]

* **目标**：生成最终答案。
* **实现**：
    1.  [cite_start]**线性化**：将患者中心图 $\mathcal{G}_{p}$ 转换成文本字符串（例如，三元组 `(subject, predicate, object)`) [cite: 245]。
    2.  [cite_start]**组合提示**：构建一个包含所有信息的最终提示（论文表8 [cite: 527] 提供了模板）。
    3.  [cite_start]**调用 LLM**：执行 $d_{m} = LLM(\mathcal{H}, \mathcal{G}_{p}, \mathcal{NP}, \mathcal{P}\mathcal{P})$ [cite: 246]。
    4.  [cite_start]**提取答案**：从 LLM 生成的响应 $d_{m}$ 中提取推荐的药物名称 [cite: 297]。

#### 3. 评估 (Evaluation)

* [cite_start]**指标**：使用 **Jaccard** 和 **F1** 分数 [cite: 257]。
* [cite_start]**方法**：在 DialMed 测试集上 [cite: 251][cite_start]，将你提取的预测药物集 $\hat{Y}^{(k)}$ 与数据集中的标准答案 $Y^{(k)}$ 进行比较 [cite: 261]。
* [cite_start]**对比**：你需要复现论文中的基线模型（Baselines），例如 I/O prompt、CoT prompt 和 KG-RAG [cite: 276, 281][cite_start]，以验证你的 GAP 实现是否达到了论文中报告的性能（见表1 [cite: 272]）。

好的，这篇论文在方法论（Methodology）和附录（Appendix）中提供了非常详细的蓝图。

如果您要复现这个名为 GAP 的框架，以下是您需要准备的核心资产和详细的实现步骤，包括论文中 1:1 提供的提示（Prompts）。

---

### 1. 准备核心资产

您需要准备以下几类工具和数据：

* **基础大语言模型 (LLMs):**
    * [cite_start]论文主要使用 **DeepSeek-V3** 作为其基础 LLM（用于提取、生成和推理） [cite: 277, 293]。
    * [cite_start]实验中也使用了 **ChatGPT-3.5-Turbo (version 1106)** [cite: 507]。

* **数据集 (Datasets):**
    * [cite_start]**DialMed**: 用于对话式药物推荐的主要数据集 [cite: 251]。这是您复现表1 结果所必需的。
    * [cite_start]**LLM as Patients**: 用于评估动态诊断访谈的辅助数据集 [cite: 253]。

* **知识图谱 (Knowledge Graphs):**
    * [cite_start]**CMeKG (中文医疗知识图谱)** [cite: 291]。
    * [cite_start]**Disease-KB** [cite: 291]。
    * [cite_start]论文还提到整合了其他关于药物、症状等的补充知识 [cite: 292]。

---

### 2. 详细实现步骤 (Algorithm 1)

[cite_start]您可以按照论文中的算法1 [cite: 185] 和第3节的描述，分四步实现该框架：

#### 步骤 1: 医疗信息提取 (Medical Information Extraction)

**目标**：从对话历史 $\mathcal{H}$ 中提取医疗概念 $\mathcal{C}$ 及其状态 $\mathcal{SV}$。

1.  [cite_start]**提取概念 (NER)**：遍历患者的每句话 $p_m$，使用 LLM 配合 **表4的提示** 来提取所有医疗概念（疾病、症状、药物） $\mathcal{C}$ [cite: 198]。
2.  [cite_start]**判断状态 (Slot-filling)**：对于提取到的每个概念 $c$，使用 LLM 配合 **表5的提示**，在一个上下文窗口内（$k=1$，即前后各一轮对话） [cite: 294, 200][cite_start]，判断该概念的状态（例如“患者自述阳性”、“既往病史：否”） [cite: 201, 469-473]。

#### 步骤 2: 患者中心图构建 (Graph Construction)

**目标**：构建患者图谱 $\mathcal{G}_{p}$ 并链接外部 KG。

1.  [cite_start]**构建图**：将步骤1中提取到的概念 $\mathcal{C}$ 和状态 $\mathcal{SV}$ 组合成一个以患者为中心的图 $\mathcal{G}_{p}$ [cite: 205]。例如：(患者) - [有症状] -> (感冒)；(感冒) - [状态] -> (患者自述阳性)。
2.  [cite_start]**实体链接**：使用简单的链接方法（如编辑距离、同义词列表） [cite: 175]，将 $\mathcal{G}_{p}$ 中的节点（如“感冒”）链接到您准备的外部知识图谱（CMeKG, Disease-KB）中的标准实体上。
3.  [cite_start]**获取邻域**：通过实体链接，您的 $\mathcal{G}_{p}$ 现在可以访问外部 KG 中的相关知识，这构成了邻域图 $\mathcal{NG_{p}}$ [cite: 176]。

#### 步骤 3: 提示生成 (Prompt Generation)

**目标**：生成邻域提示 (NP) 和基于路径的提示 (PP)。

1.  **邻域提示 (NP)**：
    * [cite_start]首先使用 LLM 判断当前对话最需要哪种类型的知识（例如，需要“治疗方案”还是“禁忌症”） [cite: 220]。
    * [cite_start]然后从邻域图 $\mathcal{NG_{p}}$ 中检索具有相应关系（relation）的 Top-k 个事实（论文中 $k=3$） [cite: 296]，作为 NP。
    * [cite_start]例如，如果患者提到“腹泻”，NP 可能是：“*ChangYanNing* 和 *Montmorillonite* 是治疗腹泻的药物” [cite: 320, 321]。

2.  **基于路径的提示 (PP)**：
    * [cite_start]**定义模式 (Schema)**：您需要预先定义一些关键的医疗推理模式 $\mathcal{S}$ [cite: 224][cite_start]。论文在 **图5 (Figure 5)** [cite: 498] 中给出了示例，例如：(pregnant) -[is: positive]-> (patient) -[has: positive]-> (symptom) -[treatment]-> (medication)。
    * [cite_start]**路径匹配**：在您的图（$\mathcal{G}_{p} + \mathcal{NG_{p}}$）中匹配这些模式 $\mathcal{S}$，以获取路径查询 $\mathcal{PS}$ [cite: 225]。
    * [cite_start]**多源检索**：使用这些路径查询 $\mathcal{PS}$ 从以下**三个来源**检索知识，并组合成 Top-k (k=3) 个 PP [cite: 229, 296]：
        1.  [cite_start]**KG 验证**：在 KG 中检查路径，例如发现“Loratadine”在“pregnancy”期间应“cautiously”使用 [cite: 232, 329]。
        2.  [cite_start]**LLM 推理**：使用 **表6的提示**，让 LLM 基于路径信息进行推理 [cite: 236]。
        3.  [cite_start]**互联网访问**：将路径查询（如“怀孕+荨麻疹+止痒”）改写为自然语言，通过搜索引擎获取知识 [cite: 241]。

#### 步骤 4: 响应生成 (Response Generation)

**目标**：生成最终的医生回复 $d_{m}$。

1.  [cite_start]**线性化**：将 $\mathcal{G}_{p}$（患者中心图）转换成文本字符串（例如，三元组列表） [cite: 245]。
2.  [cite_start]**组合提示**：构建一个最终的、包含所有信息的提示。论文使用 **表8的提示** 模板 [cite: 526]。
3.  [cite_start]**调用 LLM**：将对话历史 $\mathcal{H}$、线性化的图 $\mathcal{G}_{p}$、邻域提示 $\mathcal{NP}$ 和路径提示 $\mathcal{PP}$ 全部输入 LLM，生成最终答案 $d_{m}$ [cite: 246]。

---

### 3. 论文提供的 Prompt (1:1)

[cite_start]以下是论文附录 A.3 节中提供的提示模板 [cite: 507]，我已按原样转录（包括原文中的拼写错误，我用 `[sic]` 标注）：

#### 表 4: 医疗概念提取 (NER) 提示
[cite_start][cite: 514]
> Prompts
>
> You are a named entity recognition (NER) annotator in the medical domain. Given a piece of medical dialogue context, you are required to return the existing disease symptom/medication entities in list format as followe [sic]
>
> Output format: [""entity1"", ""entity2"", ...]
>
> The definition of disease, symptom medication: Definition
>
> [Demonstrations]
>
> Input context:
>
> Context
>
> Output result:

#### 表 5: 概念状态判断 (Slot-filling) 提示
[cite_start][cite: 519]
> Prompts
>
> You are an experienced doctor. Identify the states of the gives [sic] disease symptom based on the medical dialogue context, and return the result in JSON format
>
> main-state, candidate types come from [""patient-positive"", ""patient-negative"", ""doctor-positive"", ""doctor-negative"", ""unknown""
>
> [Descriptions of main-state types]
>
> past medical history: candidate types come from [""yes"", ""no""| [sic]
>
> [Descriptions of past medical history types]
>
> other relevant information: other information about the given disease, syruptom [sic] mentioned in the dialogue, such as duration and body parts, store the information in list format
>
> [Demonstrations]
>
> Input context:
>
> [Context]
>
> The disease/symptom:
>
> [Disease/Symptom]
>
> #Output:

#### 表 6: 用于生成 PP 的 LLM 推理提示
[cite_start][cite: 520]
> Prompts
>
> You are an experienced doctor. Please provide some effective suggestions for the following medical questions, within 50 words.
>
> Questiotr [sic] Question
>
> Suggestions:

#### 表 7: 基线 (Baseline) CoT 提示
[cite_start][cite: 522]
> Prompts
>
> You are an experienced doctor. Given a piece of medical dialogue context, you are required to recommend the medication based on patient's disease and symptom. The diseases are from Respirator, Gastroenterology, Dermatology, including: Disease]. Candidate medications are from [Medication]. You need to think step by step and generate your thoughts. Demonstrations are as follow:
>
> [Demonstrations]
>
> Input context:
>
> [Context]
>
> Now generate your thoughts and answers, please make sure the answers are from the candidate medication list:

#### 表 8: GAP 框架最终提示
[cite_start][cite: 526]
> Prompts
>
> You are an experienced doctor. Given a piece of medical dialogue context, you are required to recommend the medication based on patient's disease and symptom. The diseases are from Respirator, Gastroenterology, Dermatology, including: Disease] Candidate medications are from [Medication]. The patient-centric graph is a summary of the dialogue. The neighborhood prompts and path-based prompts can be viewed as relevant knowledge. You need to think step by step and generate your thoughts based on the context. Demonstrations are as followe [sic]
>
> [Demonstrations]
>
> Input context:
>
> [Context]
>
> Patient-centric graph:
>
> [Graph]
>
> Neighborhood Prompts:
>
> [NP]
>
> Path-hased [sic] Prompts:
>
> [PP]
>
> Now generate your thoughts and answers, please make sure the answers are from the candidate medication list:

---


