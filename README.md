# Android Demo 交付流程说明

本文档说明如何从手动触发 GitHub Action 开始，产出 Android Demo、TFLite 模型资源包，并最终交付给客户端。

适用范围：
- 当前仓库中的 `Build Android Demo` GitHub Action
- Android NER Demo
- 旧版 ALBERT NER 模型的 TFLite 导出流程

## 1. 总体流程

整套流程分为 4 步：

1. 在 GitHub Actions 手动触发 `Build Android Demo`
2. Action 下载模型并导出 TFLite
3. Action 构建 Android Demo APK，并打包资源包
4. 从 Actions Artifacts 下载产物并交付客户端

## 2. 手动触发 GitHub Action

工作流文件：
- [build-android-demo.yml](/Users/hanbing/iflytech/HanLP/.github/workflows/build-android-demo.yml)

该工作流目前仅支持手动触发：
- GitHub 仓库页面
- `Actions`
- 选择 `Build Android Demo`
- 点击 `Run workflow`

可配置参数如下：

- `quantize`
  - 是否启用量化
  - `false` 时导出 `float32`
- `quant_mode`
  - 可选值：
  - `float16`
  - `int8`
  - `hybrid-int8`
  - `full-int8`
  - `all`
- `max_seq_length`
  - 导出模型使用的最大输入长度
  - 默认 `128`

## 3. 参数选择建议

建议优先按以下规则选择：

- `float16`
  - 精度通常稳定
  - 模型体积较小
  - 适合默认交付
- `int8`
  - 权重量化
  - 一般比 `float16` 更省模型存储，但运行时收益有限
  - 精度通常也比较稳定
- `hybrid-int8`
  - 使用代表数据做校准，但不强制纯 int8 输入输出
  - 比 `full-int8` 稳定，适合需要继续压缩时尝试
- `full-int8`
  - 风险最高
  - 可能出现识别结果明显退化甚至全部为 `O`
  - 只有在确认精度可接受时才建议交付
- `all`
  - 一次导出多种精度模型并一起打包
  - 适合测试、对比和交付前评估

关于 `max_seq_length`：

- 值越大，单次推理可处理的上下文越长
- 值越大，运行时内存占用越高
- 当前 Android 端已支持超长文本自动分 chunk 推理，因此通常不必盲目把长度设得很大
- 如果客户端设备内存紧张，可优先尝试 `64` 或 `96`

## 4. Action 在做什么

工作流执行时会完成这些动作：

1. 下载 HanLP NER 模型：
   - `ner_albert_base_zh_msra_20200111_202919`
2. 下载 Hugging Face 的基础模型和 tokenizer：
   - `uer/albert-base-chinese-cluecorpussmall`
3. 调用导出脚本生成 SavedModel 和 `.tflite`
4. 将模型、`labels.txt`、`vocab.txt` 复制到 Android app 的 assets
5. 构建 `app-debug.apk`
6. 打包资源文件 zip 并上传为 Artifacts

导出脚本：
- [export_ner_tflite_legacy_albert.py](/Users/hanbing/iflytech/HanLP/tools/export_ner_tflite_legacy_albert.py)

代表数据文件：
- [rep_text_zh.txt](/Users/hanbing/iflytech/HanLP/tools/rep_text_zh.txt)

## 5. 不同模式对应的模型文件名

工作流会生成以下文件名：

- `float32` -> `ner_float32.tflite`
- `float16` -> `ner_float16.tflite`
- `int8` -> `ner_int8.tflite`
- `hybrid-int8` -> `ner_hybrid_int8.tflite`
- `full-int8` -> `ner_full_int8.tflite`

当 `quant_mode=all` 时，会同时导出：

- `ner_float16.tflite`
- `ner_int8.tflite`
- `ner_hybrid_int8.tflite`
- `ner_full_int8.tflite`

## 6. Action 产物说明

Action 完成后，Artifacts 中通常包含：

- `app-debug.apk`
  - Android Demo 安装包
- `ner_*.tflite`
  - 导出的 TFLite 模型
- `labels.txt`
  - 标签表
- `meta.json`
  - 模型元信息
- `vocab.txt`
  - tokenizer 词表
- `hanlp_ner_assets_*.zip`
  - 资源包

资源包 zip 中包含：

- 一个或多个 `.tflite`
- `labels.txt`
- `meta.json`
- `vocab.txt`

## 7. Android Demo 当前行为

Android 端关键代码：

- [MainActivity.java](/Users/hanbing/iflytech/HanLP/android/app/src/main/java/com/hanlp/nerdemo/MainActivity.java)
- [NerEngine.java](/Users/hanbing/iflytech/HanLP/android/app/src/main/java/com/hanlp/nerdemo/NerEngine.java)

当前 Demo 支持：

- 自动识别 assets 中有哪些模型文件
- 在界面上切换不同精度模型
- 文本超过模型最大长度时自动分 chunk 推理
- 优先在中英文常见标点处分割
- 每个 chunk 分别推理后合并实体结果
- 在结果区附带当前模型的调试信息

这意味着：

- 即使 `max_seq_length` 比较小，也可以处理更长文本
- 但长文本会被拆成多个片段，推理耗时会增加
- chunk 边界处的实体连续性可能略受影响

## 8. 推荐交付方式

如果是面向客户端交付，推荐分两种情况：

### 方案 A：交付单一稳定版本

适用于：
- 客户端不需要自己切换精度
- 只需要一个可用版本

建议交付：

- `app-debug.apk` 或正式签名 APK
- 一个资源包 zip

推荐精度：

- 首选 `int8`
- 如需兼顾稳妥性可选择 `float16`
- 不建议默认交付 `full-int8`

### 方案 B：交付多精度对比版本

适用于：
- 客户端需要在设备上对比速度、内存和精度
- 客户端希望自行选型

建议触发方式：

- `quant_mode=all`

建议交付：

- `app-debug.apk`
- `hanlp_ner_assets_all.zip`

说明：

- Demo 中可直接切换不同精度
- 便于现场测试
- 便于客户端根据设备情况做取舍

## 9. 推荐给客户端的交付内容

最小交付集：

- APK
- 对应资源包 zip
- 一页简要说明

建议说明内容至少包含：

- 当前模型用途：中文 NER
- 当前精度：例如 `float16`
- 最大输入长度：例如 `128`
- 长文本处理方式：自动 chunk
- 已知限制：
  - `full-int8` 可能精度不稳定
  - chunk 边界可能影响少量实体

## 10. 如何验证交付物

交付前至少做一次以下检查：

1. 安装 APK
2. 用一条短文本验证能正常识别实体
3. 用一条长文本验证 chunk 逻辑正常
4. 切换不同精度模型验证不会崩溃
5. 检查结果区调试信息，确认实际加载的是预期模型文件

建议使用以下文本做基础回归：

```text
上海华安工业（集团）公司董事长谭旭光和秘书张晚霞来到美国纽约现代艺术博物馆参观。
```

## 11. 当前实践建议

基于当前验证情况，建议如下：

- 默认交付优先选 `int8`
- 如果需要更稳妥的兼容性，可回退到 `float16`
- 如果需要继续压缩，可尝试 `hybrid-int8`
- `full-int8` 只有在专项验证通过后再交付

如果目标是客户端正式接入，而不是 Demo 演示，还建议额外补充：

- 正式签名 APK
- 一份接口/资源接入说明
- 一份精度与性能对比表
