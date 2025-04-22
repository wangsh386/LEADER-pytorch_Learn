import numpy as np

##########################################
# LLAMA模型基础训练数据处理器
##########################################
class llama_train(object):
    def __init__(self, data_args, model_args, tokenizer):
        """初始化训练数据处理流程
        Args:
            data_args:  数据配置参数（如最大输入长度）
            model_args: 模型配置参数（当前版本未使用，预留接口） 
            tokenizer:  LLAMA分词器"""
        
        # 参数绑定
        self.data_args = data_args  # <注释说明>：控制最大输入/输出长度等关键参数
        self.model_args = model_args 
        # 字段定义
        self.prompt_column = "input"   # 输入问题字段名 
        self.response_column = "target"# 输出回答字段名
        self.history_column = None     # 多轮对话历史字段（若有则为列表）
        self.tokenizer = tokenizer     # 分词器实例

    def __call__(self, examples):
        """将原始数据批量转换为模型输入格式
        核心流程：
          1. 构建多轮对话提示词 
          2. 编码prompt与response
          3. 拼接输入并生成labels
          4. 处理填充与掩码"""
        max_seq_length = self.data_args.max_source_length + self.data_args.max_target_length
        model_inputs = {"input_ids": [], "labels": []}  # <注释说明>：与transformers库标准输入对齐

        for i in range(len(examples[self.prompt_column])):
            # 跳过空数据
            if not examples[self.prompt_column][i] or not examples[self.response_column][i]:
                continue 

            # Step1: 多轮对话拼接（若存在历史）
            query, answer = examples[self.prompt_column][i], examples[self.response_column][i]
            if self.history_column:  # <注释说明>：多轮对话模板为"[Round X]问:...答:..."
                prompt, history = "", examples[self.history_column][i]
                for turn_idx, (old_q, resp) in enumerate(history):
                    prompt += f"[Round {turn_idx}]\n问：{old_q}\n答：{resp}\n"
                prompt += f"[Round {len(history)}]\n问：{query}\n答："
            else:  # 单轮对话直接使用query
                prompt = query

            # Step2: 编码文本（不自动添加特殊token）
            a_ids = self.tokenizer.encode(prompt, add_special_tokens=False)  # prompt部分编码
            b_ids = self.tokenizer.encode(answer, add_special_tokens=False)  # response部分编码

            # Step3: 文本截断（保持前max_source_length-1个token，预留位置给EOS等）
            a_ids = a_ids[:self.data_args.max_source_length - 1] if len(a_ids) > self.max_source_length else a_ids
            b_ids = b_ids[:self.data_args.max_target_length - 2] if len(b_ids) > self.max_target_length else b_ids

            # Step4: 构建完整输入（拼接prompt+response+EOS）
            input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]  
            # <注释说明>：labels中prompt部分用pad填充，response保留真实值用于损失计算
            labels = [self.tokenizer.pad_token_id]*len(a_ids) + b_ids + [self.tokenizer.eos_token_id]
            
            # Step5：处理pad与损失掩码（是否忽略pad位置的loss）
            pad_len = max(0, max_seq_length - len(input_ids))
            input_ids += [self.tokenizer.pad_token_id] * pad_len  # pad到统一长度
            labels += [self.tokenizer.pad_token_id] * pad_len
            if self.data_args.ignore_pad_token_for_loss:
                labels = [(-100 if l == self.tokenizer.pad_token_id else l) for l in labels]  # <关键点>-100为PyTorch忽略的损失值
            
            # Step6：记录batch数据
            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)

        return model_inputs  # 格式: {"input_ids": [[t1,t2...]], "labels": [[lab1,lab2...]]}

##########################################
# 推理评估阶段数据处理器
##########################################
class llama_eval(object):
    def __init__(self, data_args, model_args, tokenizer):
        """评估数据处理（相比训练简化历史处理与标签格式）"""
        # ...（初始化类似训练类）...

    def __call__(self, examples):
        inputs, targets = [], []
        # 步骤1：构建输入prompt（逻辑同训练）
        for i in range(len(examples[self.prompt_column])):
            # ...（拼接多轮对话逻辑）...
            inputs.append(prompt)  # 生成推理用提示词列表
            targets.append(examples[self.response_column][i] or "filled in !")  # 占位符处理

        # 步骤2：批量编码输入，无需拼接响应（模型自回归生成）
        model_inputs = self.tokenizer(
            inputs, 
            max_length=self.data_args.max_source_length, 
            truncation=True, 
            padding=True
        )
        # 步骤3：独立编码目标文本（用于计算BLEU等指标）
        labels = self.tokenizer(
            text_target=targets, 
            max_length=self.data_args.max_target_length, 
            truncation=True
        )
        # 处理填充符的loss掩码（同训练）
        if self.data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] 
                for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs  # 格式与训练一致


##########################################
# 分类任务适应版处理器（如药物代码预测）
##########################################
class llama_train_cls(object):
    def __init__(self, data_args, model_args, tokenizer, ehr_tokenizer):
        """适应分类任务的改进处理器（如用药建议）
        核心差异：
          1. 标签本身为药物编码列表而非文本 
          2. 输出改为多标签one-hot向量"""
        # 新增医疗专用分词器（例如将药物编码转换为ID）
        self.ehr_tokenizer = ehr_tokenizer  # <注释说明>：如EHRTokenizer可转换ATC编码
        self.response_column = "drug_code"  # 标签列存储药物编码

    def __call__(self, examples):
        model_inputs = {"input_ids": [], "labels": []}
        med_voc_size = len(self.ehr_tokenizer.med_voc.word2idx)  # 药物词表大小

        for i in range(len(examples[self.prompt_column])):
            # 输入处理逻辑同训练（拼接prompt）...
            input_ids = ...  # 同训练阶段

            # 标签转换核心代码
            answer_codes = examples[self.response_column][i]  # 原始药物编码列表 ["J01EA04", ...]
            label_indices = self.ehr_tokenizer.convert_med_tokens_to_ids(answer_codes)  # 转为ID列表 [23, 45, ...]
            
            # 生成多标签二值向量（如多药联合使用场景）
            labels = np.zeros(med_voc_size, dtype=np.float32)  # 全0向量
            labels[label_indices] = 1  # 对应药物位置设为1
            # <注释说明>：模型最后需使用sigmoid+BCE损失或softmax+CE损失

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)
        return model_inputs  # labels形状变为 [batch_size, med_voc_size]


##########################################
# DPO（直接偏好优化）数据处理 
##########################################
class llama_dpo_cls(object):
    def __init__(self, data_args, model_args, tokenizer, ehr_tokenizer):
        """DPO训练数据处理
        核心功能：处理偏好对数据（正例：合理用药，反例：错误用药）"""
        self.positive_column = "positive"  # 正样本药物编码
        self.negative_column = "negative"  # 负样本药物编码

    def __call__(self, examples):
        model_inputs = {"prompt_ids": [], "chosen_ids": [], "rejected_ids": []}
        # 遍历构建偏好对...
        for i in range(len(...)):
            pos_codes = examples[self.positive_column][i] 
            neg_codes = examples[self.negative_column][i]
            
            # 生成正/负标签的one-hot向量
            pos_labels = np.zeros(med_voc_size)
            pos_labels[self.ehr_tokenizer.convert_med_tokens_to_ids(pos_codes)] = 1
            # 同理生成neg_labels...

            # DPO需模型的chosen和rejected输出logits进行比较
            # 此处示例假设通过one-hot对比，实际需适配具体DPO实现
            model_inputs["chosen_ids"].append(pos_labels)
            model_inputs["rejected_ids"].append(neg_labels)
        return model_inputs
