# 版权声明：此代码基于Apache License 2.0许可
# 主要功能：针对序列到序列任务和医疗推荐任务定制的Trainer类

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, PreTrainedModel, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollator
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction, PredictionOutput
from transformers.utils import logging

logger = logging.get_logger(__name__)

class Seq2SeqTrainer(Trainer):
    """继承自Hugging Face Trainer的序列到序列任务专用训练器
    
    扩展功能：
    1. 支持生成式任务的评估和预测
    2. 自动处理生成参数（如beam search）
    3. 处理生成结果与标签的对齐
    """

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs
    ) -> Dict[str, float]:
        """扩展评估方法，整合生成参数
        
        参数说明：
        - eval_dataset: 评估数据集，默认为训练器初始化时设定的验证集
        - ignore_keys: 模型输出中需要忽略的键（如辅助输出）
        - metric_key_prefix: 评估指标前缀（如'eval'会生成'eval_bleu'等指标）
        - gen_kwargs: 生成参数（max_length, num_beams等）
        
        流程：
        1. 配置生成参数（优先使用传入参数，其次使用训练参数默认值）
        2. 调用父类评估方法
        """
        # 生成参数预处理
        gen_kwargs = gen_kwargs.copy()
        # 设置默认最大生成长度
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        # 设置默认beam数
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] 
            if gen_kwargs.get("num_beams") is not None 
            else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs  # 保存生成参数供后续使用

        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        **gen_kwargs
    ) -> PredictionOutput:
        """预测方法，生成文本结果并计算指标
        
        参数说明：
        - test_dataset: 测试数据集
        - metric_key_prefix: 测试指标前缀
        - gen_kwargs: 生成参数
        
        返回：
        PredictionOutput对象包含：
        - predictions: 模型预测结果（numpy数组）
        - label_ids: 真实标签（如果数据集包含）
        - metrics: 评估指标字典
        """
        # 生成参数处理逻辑同evaluate
        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] 
            if gen_kwargs.get("num_beams") is not None 
            else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """单步预测核心逻辑
        
        参数说明：
        - model: 当前模型
        - inputs: 输入张量字典
        - prediction_loss_only: 是否仅返回loss
        - ignore_keys: 需要忽略的模型输出键
        
        返回：
        (loss, generated_tokens, labels) 元组
        
        流程：
        1. 准备生成参数
        2. 调用model.generate生成文本
        3. 对齐生成结果与标签长度
        4. 处理特殊设备情况（如DeepSpeed Zero-3）
        """
        # 如果不需要生成或只需loss，调用父类方法
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)  # 将输入转移到正确设备

        # 生成参数配置
        gen_kwargs = self._gen_kwargs.copy()
        # 配置同步GPU参数（DeepSpeed Zero-3特殊处理）
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = gen_kwargs.get("synced_gpus", default_synced_gpus)

        # 准备注意力掩码等输入
        for key in ["attention_mask", "position_ids", "global_attention_mask"]:
            if key in inputs:
                gen_kwargs[key] = inputs.get(key)

        # 确定生成输入（适配不同encoder-decoder模型）
        if hasattr(model, "encoder") and model.encoder.main_input_name != model.main_input_name:
            generation_inputs = inputs[model.encoder.main_input_name]
        else:
            generation_inputs = inputs[model.main_input_name]

        # 执行生成
        gen_kwargs["input_ids"] = generation_inputs
        generated_tokens = model.generate(**gen_kwargs)
        
        # 截断生成结果（移除输入部分）
        generated_tokens = generated_tokens[:, generation_inputs.size()[-1]:]

        # 填充生成结果到指定长度
        if gen_kwargs.get("max_length") and generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        elif gen_kwargs.get("max_new_tokens"):
            target_length = gen_kwargs["max_new_tokens"] + 1
            if generated_tokens.shape[-1] < target_length:
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, target_length)

        # 处理标签
        labels = None
        if has_labels:
            labels = inputs["labels"]
            # 对齐标签长度
            if gen_kwargs.get("max_length") and labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
            elif gen_kwargs.get("max_new_tokens"):
                target_length = gen_kwargs["max_new_tokens"] + 1
                if labels.shape[-1] < target_length:
                    labels = self._pad_tensors_to_max_len(labels, target_length)

        return (None, generated_tokens, labels)  # loss设为None

    def _pad_tensors_to_max_len(self, tensor: torch.Tensor, max_length: int) -> torch.Tensor:
        """将张量填充到指定长度
        
        参数：
        - tensor: 需要填充的张量 (batch_size, seq_len)
        - max_length: 目标长度
        
        返回：
        padded_tensor: 填充后的张量 (batch_size, max_length)
        """
        # 确定填充token ID
        pad_token_id = (
            self.tokenizer.pad_token_id 
            if self.tokenizer is not None 
            else self.model.config.pad_token_id
        )
        if pad_token_id is None:
            raise ValueError("需要设置pad_token_id以进行填充")

        # 创建填充模板
        padded_shape = (tensor.size(0), max_length)
        padded_tensor = pad_token_id * torch.ones(padded_shape, dtype=tensor.dtype, device=tensor.device)
        
        # 复制原始数据
        padded_tensor[:, : tensor.size(-1)] = tensor
        return padded_tensor


class MedRecTrainer(Trainer):
    """医疗推荐系统专用训练器
    
    功能特点：
    1. 适配结构化输出（非生成式任务）
    2. 返回隐藏状态用于后续分析
    3. 优化内存管理（仅返回必要输出）
    """

    def prediction_step(
        self, 
        model: nn.Module, 
        inputs: Dict[str, torch.Tensor], 
        prediction_loss_only: bool, 
        ignore_keys: List[str]
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """医疗推荐预测步骤
        
        参数说明：
        - model: 医疗推荐模型
        - inputs: 包含患者病史、用药记录等的输入张量
        - prediction_loss_only: 是否仅计算损失
        
        返回：
        (loss, logits, hidden_states) 元组
        
        流程：
        1. 准备模型输入
        2. 获取模型输出（包含隐藏状态）
        3. 返回关键信息，避免内存溢出
        """
        has_labels = "labels" in inputs

        # 准备输入
        inputs = self._prepare_inputs(inputs)
        gen_kwargs = {
            "attention_mask": inputs.get("attention_mask"),
            "position_ids": inputs.get("position_ids"),
            "global_attention_mask": inputs.get("global_attention_mask"),
            "input_ids": inputs[model.main_input_name],
            "labels": inputs.get("labels")
        }

        # 前向传播
        outputs = model(**gen_kwargs)
        
        # 提取关键输出
        loss = None
        logits = outputs.get("logits")
        hidden_states = outputs.get("hidden_states")

        return (loss, logits, hidden_states)
