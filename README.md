# sohu2022-nlp-rank1

2022搜狐校园算法大赛NLP赛道第一名开源方案（实验代码）

[方案介绍文章](https://zhuanlan.zhihu.com/p/533808475)

该代码使用pytorch-lightning框架进行编写。**注意：该代码为我本人在初赛阶段实验和迭代使用的代码，并非用于复赛和决赛提交的代码，有部分trick没有加入，效果会比最终提交代码稍差。**

核心代码段：

`datamodule.py`的输入构造部分

```python
def _setup(self, data):
    output = []
    for item in tqdm(data):
        output_item = {}
        text = item["content"]
        if not text or not item["entity"]:
            continue
        prompt = "".join([f"{entity}{self.mask_symbol}" for entity in item["entity"]])
        inputs = self.tokenizer.__call__(text=text, text_pair=prompt, add_special_tokens=True, max_length=self.hparams.max_length, truncation="only_first")
        inputs["is_masked"] = [int(i == self.tokenizer.mask_token_id) for i in inputs["input_ids"]]
        inputs["first_mask"] = [int(i == 0) for i in inputs["token_type_ids"]]
        output_item["inputs"] = inputs
        if isinstance(item["entity"], dict):
            labels = list(map(lambda x: x + 2, item["entity"].values()))
            output_item["labels"] = labels
        output.append(output_item)
```

`model.py`的`forward`部分

```python
def forward(self, inputs, output_hidden_states=False):
    is_masked = inputs['is_masked'].bool()
    first_mask = inputs.get('first_mask', None)
    inputs = {k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}
    backbone_outputs = self.xlnet(**inputs, output_hidden_states=True)
    masked_outputs = backbone_outputs.last_hidden_state[is_masked]
    ...
    logits = self.classifier(masked_outputs)
    if not output_hidden_states:
        return logits
    ...
```
