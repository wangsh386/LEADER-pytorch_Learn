# 导入所需的库
from typing import Optional, List, Union, Tuple
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss  # 导入用于损失计算的函数
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaModel, LlamaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutputWithPast  # 导入模型输出格式

# 定义LlamaForMedRec模型，继承自LlamaPreTrainedModel
class LlamaForMedRec(LlamaPreTrainedModel):

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        """
        初始化LlamaForMedRec模型，继承自LlamaPreTrainedModel
        参数:
        - config: 配置文件，用于初始化模型
        - med_voc: 医疗领域的词汇大小（即标签数）
        """
        super().__init__(config, *inputs, **kwargs)  # 调用父类初始化方法
        self.model = LlamaModel(config)  # Llama的主模型
        self.config = config

        # 医疗嵌入适配器：用于处理不同类型的医疗嵌入（诊断、手术、药物）
        self.med_voc = kwargs.pop("med_voc")  # 从kwargs中获取医疗词汇表的大小
        self.cls_head = nn.Linear(config.hidden_size, self.med_voc, bias=False)  # 分类头，用于映射到药物类别空间

        # 初始化权重和应用最终的处理
        self.post_init()  # 调用父类的后初始化方法

    def get_input_embeddings(self):
        """
        获取模型的输入嵌入层。
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """
        设置模型的输入嵌入层。
        """
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的token ids
        attention_mask: Optional[torch.Tensor] = None,  # 注意力mask
        position_ids: Optional[torch.LongTensor] = None,  # 位置编码
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 上下文历史
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入嵌入
        labels: Optional[torch.LongTensor] = None,  # 目标标签
        use_cache: Optional[bool] = None,  # 是否缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏层状态
        return_dict: Optional[bool] = None,  # 是否以字典形式返回
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        """
        模型的前向传播函数，执行输入的处理、嵌入注入、Transformer计算和分类任务。
        """
        # 默认返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取Transformer的输出
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取Transformer的隐藏状态（隐藏层输出）
        hidden_states = transformer_outputs[0]  # (batch_size, seq_len, hidden_size)
        
        # 将隐藏状态传递到分类头，得到每个类别的预测logits
        logits = self.cls_head(hidden_states)  # (batch_size, seq_len, med_voc)

        # 如果存在input_ids，则获取批次大小
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        # 检查是否存在padding token，确保batch_size为1时能正确处理
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # 计算每个样本的序列长度（去除padding的部分）
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1

        # 选择每个样本的最后一个有效token的logits
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        # 如果提供了labels，则计算损失
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)  # 将标签移动到logits所在的设备

            # 使用BCEWithLogitsLoss计算损失
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(pooled_logits, labels.float())  # 使用BCE计算多标签分类的损失

        # 如果不返回字典，则以元组形式返回
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果返回字典格式的输出
        transformer_outputs.hidden_states = hidden_states[torch.arange(batch_size, device=logits.device), sequence_lengths]  # 获取隐藏状态
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
