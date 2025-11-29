# MixTokenizer Copilot 指南

欢迎来到 MixTokenizer！本项目旨在将一个基础分词器（如 Qwen2）与一个针对新语言（使用私有 Unicode 字符表示）的分词器进行混合。这种混合是通过将新语言的词汇映射到基础分词器中低频词元的组合来实现的。

## 核心架构

项目的核心思想是创建一个动态的 `MixTokenizer` 类，该类继承自一个标准的 `transformers` 分词器类（例如 `Qwen2Tokenizer`）。这个混合分词器能够处理标准文本和我们自定义的新语言文本。

1.  **字符映射**:
    *   `MixTokenizer/mapping.py` 中的 `PrivateUnicodeMapper` 负责将源语言（如中文）的字符映射到私有使用区（PUA）的 Unicode 字符。
    *   这个映射是训练过程的第一步，由 `config.yaml` 中的 `mapping` 部分配置。

2.  **分词器训练**:
    *   `scripts/train.py` 是主要的训练脚本。它读取 `config.yaml`，然后执行以下步骤：
        1.  **准备映射**: 创建或加载字符映射。
        2.  **创建新语言分词器**: 为 PUA 字符创建一个分词器。
        3.  **频率计数**: 统计基础分词器在语料库中的词元频率。
        4.  **训练 `MixTokenizer`**: `MixTokenizer/tokenizer_train.py` 中的 `get_mix_tokenizer` 会创建一个继承自基础分词器的新类。它的 `.train()` 方法会使用低频词元为新语言的每个词元创建组合映射。
        5.  **生成注入文件**: 训练完成后，脚本会将所有必需的配置（如映射、新分词器词汇表）和动态生成的 `tokenizer.py` 保存到 `config.yaml` 中指定的 `output_dir`（默认为 `mix/`）。

3.  **模型注入**:
    *   `scripts/install_mix.py` 脚本将训练产物（即 `mix/` 目录）复制到目标 Hugging Face 模型目录中。
    *   它会修改模型的 `tokenizer_config.json`，将 `tokenizer_class` 指向 `MixTokenizer`，并设置 `auto_map` 以便 `AutoTokenizer.from_pretrained` 能够加载它（需要 `trust_remote_code=True`）。

4.  **核心 C++ 逻辑**:
    *   `MixTokenizer/core/cpp_core.cpp` 包含一个使用 pybind11 编写的高性能 C++ 扩展，用于加速新语言词元到基础词元组合的转换。这对于在运行时保持高性能至关重要。

## 关键工作流程

### 1. 训练一个新的 `MixTokenizer`

这是最常见的任务。

1.  **配置**: 编辑 `config.yaml` 文件。你需要指定：
    *   `model_name_or_path`: 你想要扩展的基础模型。
    *   `mapping`: 要映射的字符范围。
    *   `mix_trained`: 训练数据 (`doc_path`) 或预计算的频率统计 (`counter_path`)，以及组合级别 (`level`)。
2.  **运行训练**:
    ```bash
    python scripts/train.py
    ```
    这将根据配置在 `mix/` 目录中生成所有必需的文件。

### 2. 将 `MixTokenizer` 安装到模型中

1.  **运行安装脚本**:
    ```bash
    python scripts/install_mix.py /path/to/your/model/ --mix_dir mix/
    ```
    这会将 `mix/` 目录复制到模型目录并更新 `tokenizer_config.json`。

### 3. 运行测试

确保你的环境已经安装了开发依赖：

```bash
pip install -e .[dev]
pytest -s tests/
```

## 项目约定

*   **动态类生成**: `MixTokenizer` 不是一个静态定义的类。它是在运行时由 `get_mix_tokenizer` 函数动态创建的，继承自 `config.yaml` 中指定的基础分词器类。
*   **配置驱动**: 整个训练过程由 `config.yaml` 文件驱动。对该文件的修改会直接影响训练结果。
*   **注入机制**: 本项目不直接修改 `transformers` 库，而是通过将一个文件夹（默认为 `mix/`）注入到现有模型目录中来工作。该文件夹包含了自定义逻辑和配置。
*   **性能**: 性能关键部分（词元转换）已卸载到 C++ 中。如果你需要修改这部分逻辑，请编辑 `MixTokenizer/core/cpp_core.cpp` 并重新编译项目。
