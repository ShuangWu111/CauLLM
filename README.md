# Causality-Enhanced LLM Fine-tuning Method for Multimodal Time Series Classification

Shuang Wu, Yingwei Zhang, Haopeng Sun.

## Get Start

1. 修改路径
cd Classification/src/models/caugpt4ts.py
修改 self.gpt2 = AutoModel.from_pretrained('/home/wushuang23s/Classification/src/models/gpt2', output_attentions=True, output_hidden_states=True) 中的路径为gpt2模型参数文件的路径

2. Train and Test
cd Classification
根据选择的数据集运行不同的脚本，例如选择数据集SelfRegulationSCP1，则运行以下命令：
bash ./scripts/SelfRegulationSCP1.sh


