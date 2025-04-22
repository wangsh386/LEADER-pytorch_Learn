# -*- coding: utf-8 -*-
# 必要依赖库：transformers, numpy, dill
import numpy as np


class llama_train(object):
    """LLaMA模型训练数据处理器（生成式任务，如对话生成）
    
    功能：
    1. 处理单轮/多轮对话格式
    2. 拼接prompt和response并添加特殊token
    3. 生成带掩码的labels用于CausalLM训练
    """
    
    def __init__(self, data_args, model_args, tokenizer) -> None:
        """初始化参数
        Args:
            data_args (DataArguments): 包含max_source_length等参数的数据配置对象
            model_args (ModelArguments): 模型配置参数（当前未使用，为后续扩展保留）
            tokenizer (LlamaTokenizer): LLaMA分词器实例
        """
        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = "input"    # 输入文本字段名（如"question"）
        self.response_column = "target" # 输出文本字段名（如"answer"）
        self.history_column = None      # 多轮对话历史字段名（可选）
        self.tokenizer = tokenizer      # 分词器实例

    def __call__(self, examples):
        """处理批量样本，生成模型输入
        Args:
            examples (dict): 原始数据批次，格式如 {"input": ["q1", "q2"], "target": ["a1", "a2"]}
        Returns:
            dict: 包含input_ids和labels的字典，适配transformers.Trainer
        """
        max_seq_length = self.data_args.max_source_length + self.data_args.max_target_length
        model_inputs = {
            "input_ids": [],
            "labels": [],
        }

        # 逐条处理样本
        for i in range(len(examples[self.prompt_column])):
            # 跳过空数据
            if not examples[self.prompt_column][i] or not examples[self.response_column][i]:
                continue

            # 获取当前样本的query和answer
            query, answer = examples[self.prompt_column][i], examples[self.response_column][i]

            # 构建多轮对话prompt（如果存在历史）
            if self.history_column is None:
                prompt = query  # 单轮对话直接使用query
            else:
                prompt = ""
                history = examples[self.history_column][i]
                # 按轮次拼接历史对话
                for turn_idx, (old_query, response) in enumerate(history):
                    prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                # 添加当前轮的问题
                prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

            # 编码prompt和answer（不自动添加特殊token）
            a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)  # prompt部分
            b_ids = self.tokenizer.encode(text=answer, add_special_tokens=False)  # response部分

            # 截断超长文本（保留头部信息）
            if len(a_ids) > self.data_args.max_source_length - 1:
                a_ids = a_ids[: self.data_args.max_source_length - 1]
            if len(b_ids) > self.data_args.max_target_length - 2:
                b_ids = b_ids[: self.data_args.max_target_length - 2]

            # 构建完整输入（拼接prompt+response+EOS）
            input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]
            
            # 生成labels（prompt部分用pad填充，response部分保留真实token）
            context_length = len(a_ids)
            labels = [self.tokenizer.pad_token_id] * context_length + b_ids + [self.tokenizer.eos_token_id]

            # 处理填充到固定长度
            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [self.tokenizer.pad_token_id] * pad_len

            # 可选：将pad位置的label设为-100（忽略损失计算）
            if self.data_args.ignore_pad_token_for_loss:
                labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

            # 添加到batch
            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)

        return model_inputs


class llama_eval(object):
    """LLaMA模型评估数据处理器（生成式任务）"""
    
    def __init__(self, data_args, model_args, tokenizer) -> None:
        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = "input"
        self.response_column = "target"
        self.history_column = None
        self.tokenizer = tokenizer

    def __call__(self, examples):
        """处理逻辑与训练类似，但保留原始target用于指标计算"""
        max_target_length = self.data_args.max_target_length
        inputs, targets = [], []

        # 构建输入和参考答案
        for i in range(len(examples[self.prompt_column])):
            # 处理空target的情况
            if not examples[self.response_column][i]:
                targets.append("filled in !")  # 占位符
            else:
                targets.append(examples[self.response_column][i])

            # 构建prompt（同训练逻辑）
            if examples[self.prompt_column][i]:
                query = examples[self.prompt_column][i]
                if self.history_column is None or len(examples[self.history_column][i]) == 0:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[self.history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
                inputs.append(prompt)

        # 批量编码输入
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.data_args.max_source_length,
            truncation=True,
            padding=True
        )
        
        # 编码target（用于计算BLEU等指标）
        labels = self.tokenizer(
            text_target=targets,
            max_length=max_target_length,
            truncation=True
        )

        # 处理pad token的损失掩码
        if self.data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] 
                for label in labels["input_ids"]
            ]
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


class llama_train_cls(object):
    """LLaMA分类任务训练处理器（如药物代码预测）"""
    
    def __init__(self, data_args, model_args, tokenizer, ehr_tokenizer) -> None:
        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = "input"      # 输入文本字段
        self.response_column = "drug_code"# 药物编码字段
        self.history_column = None
        self.tokenizer = tokenizer
        self.ehr_tokenizer = ehr_tokenizer  # 医疗编码专用分词器

    def __call__(self, examples):
        """生成多分类标签（one-hot格式）"""
        max_seq_length = self.data_args.max_source_length + self.data_args.max_target_length
        model_inputs = {
            "input_ids": [],
            "labels": [],
        }

        for i in range(len(examples[self.prompt_column])):
            if examples[self.prompt_column][i] and examples[self.response_column][i]:
                query, answer = examples[self.prompt_column][i], examples[self.response_column][i]

                # 构建prompt（同生成式任务）
                if self.history_column is None:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[self.history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

                # 编码prompt
                a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)
                if len(a_ids) > self.data_args.max_source_length - 1:
                    a_ids = a_ids[: self.data_args.max_source_length - 1]

                # 构建输入（添加EOS）
                input_ids = a_ids + [self.tokenizer.eos_token_id]
                
                # 生成多分类标签（例如预测多个药物）
                label_index = self.ehr_tokenizer.convert_med_tokens_to_ids(answer)  # 转换编码为ID
                med_voc_size = len(self.ehr_tokenizer.med_voc.word2idx)
                labels = np.zeros((med_voc_size))
                labels[label_index] = 1  # 多标签设置为1

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)

        return model_inputs


class llama_eval_cls(object):
    """LLaMA分类任务评估处理器"""
    
    def __init__(self, data_args, model_args, tokenizer, ehr_tokenizer) -> None:
        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = "input"
        self.response_column = "drug_code"
        self.history_column = None
        self.tokenizer = tokenizer
        self.ehr_tokenizer = ehr_tokenizer

    def __call__(self, examples):
        max_target_length = self.data_args.max_target_length
        inputs, targets = [], []

        # 处理每个样本
        for i in range(len(examples[self.prompt_column])):
            # 生成one-hot标签
            label_index = self.ehr_tokenizer.convert_med_tokens_to_ids(examples[self.response_column][i])
            med_voc_size = len(self.ehr_tokenizer.med_voc.word2idx)
            labels = np.zeros((med_voc_size))
            labels[label_index] = 1
            targets.append(labels)

            # 构建prompt（同训练逻辑）
            if examples[self.prompt_column][i]:
                query = examples[self.prompt_column][i]
                if self.history_column is None or len(examples[self.history_column][i]) == 0:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[self.history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
                inputs.append(prompt)

        # 批量编码输入
        inputs = [inp for inp in inputs]
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.data_args.max_source_length,
            truncation=True,
            padding=True
        )
        model_inputs["labels"] = targets  # 直接使用预先生成的标签

        return model_inputs


class llama_dpo_cls(object):
    """DPO（直接偏好优化）数据处理"""
    
    def __init__(self, data_args, model_args, tokenizer, ehr_tokenizer) -> None:
        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = "input"       # 输入字段
        self.positive_column = "positive"  # 正样本药物编码
        self.negative_column = "negative"  # 负样本药物编码
        self.history_column = None
        self.tokenizer = tokenizer
        self.ehr_tokenizer = ehr_tokenizer

    def __call__(self, examples):
        """生成偏好对数据"""
        max_seq_length = self.data_args.max_source_length + self.data_args.max_target_length
        model_inputs = {
            "prompt_ids": [],   # 输入prompt
            "chosen_ids": [],   # 优选答案（正样本）
            "rejected_ids": [], # 劣选答案（负样本）
        }

        for i in range(len(examples[self.prompt_column])):
            if examples[self.prompt_column][i] and examples[self.positive_column][i]:
                query = examples[self.prompt_column][i]
                positive, negative = examples[self.positive_column][i], examples[self.negative_column][i]

                # 构建prompt（同训练逻辑）
                if self.history_column is None:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[self.history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

                # 编码prompt
                a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)
                if len(a_ids) > self.data_args.max_source_length - 1:
                    a_ids = a_ids[: self.data_args.max_source_length - 1]

                # 构建输入
                input_ids = a_ids + [self.tokenizer.eos_token_id]

                # 生成正/负样本标签
                med_voc_size = len(self.ehr_tokenizer.med_voc.word2idx)
                positive_labels = np.zeros((med_voc_size))
                negative_labels = np.zeros((med_voc_size))
                positive_labels[self.ehr_tokenizer.convert_med_tokens_to_ids(positive)] = 1
                negative_labels[self.ehr_tokenizer.convert_med_tokens_to_ids(negative)] = 1

                # 添加到batch
                model_inputs["prompt_ids"].append(input_ids)
                model_inputs["chosen_ids"].append(positive_labels)
                model_inputs["rejected_ids"].append(negative_labels)

        return model_inputs
