# -*- coding: utf-8 -*-
# 设置可见GPU设备（4-7号卡）
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import json
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

# 导入LoRA相关模块
from llm.peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PeftModel,
)

# 导入Transformers相关组件
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaForSequenceClassification
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer, HfArgumentParser, Seq2SeqTrainingArguments
from transformers import AutoModel, AutoTokenizer
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

# 导入自定义模块
from llm.llama import LlamaForMedRec  # 医疗定制版LLaMA
from llm.trainer_seq2seq import MedRecTrainer  # 定制训练器
from llm.lora_cls import PeftModelForCLS  # 分类任务适配
from llm.arguments import DataTrainingArguments, ModelArguments  # 参数配置
from llm.data_processor.llama import llama_train_cls, llama_eval_cls  # 数据预处理
from llm.data_processor.collator import LongestSequenceCollator  # 数据批处理
from generators.data import Voc, EHRTokenizer  # 电子健康记录处理
from evaluate import evaluate_jsonlines  # 评估指标
import time

# 自定义模型保存回调（适配PeftModel）
class SavePeftModelCallback(TrainerCallback):
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """自定义保存逻辑，避免保存完整模型权重"""
        if state.is_world_process_zero:  # 主进程执行
            print('+++++++++++++++++保存检查点回调++++++++++++++++')
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )
            # 只保存适配器权重
            kwargs["model"].save_pretrained(checkpoint_folder)
            
            # 删除自动生成的冗余文件
            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            return control

def train():
    # 参数解析（模型参数/数据参数/训练参数）
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    device_map = "auto"  # 自动分配GPU设备

    # 加载医疗词表（诊断/治疗/药物）
    voc_dir = "data/mimic3/handled/voc_final.pkl"
    ehr_tokenizer = EHRTokenizer(voc_dir)  # 包含医疗编码转换功能

    # 模型加载 ============================================================
    print("********************** 加载基础模型 **********************")
    # 加载医疗定制版LLaMA（扩展了药物分类头）
    model = LlamaForMedRec.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,  # LLaMA模型路径
        med_voc=len(ehr_tokenizer.med_voc.word2idx)  # 药物词表大小
    ).half().cuda()  # 半精度加载到GPU

    # LoRA微调配置 =======================================================
    if model_args.peft_path is not None:  # 测试模式加载现有适配器
        if training_args.resume_from_checkpoint:  # 恢复训练
            model = PeftModelForCLS.from_pretrained(model, model_args.peft_path, is_trainable=True)
        else:  # 仅推理
            model = PeftModelForCLS.from_pretrained(model, model_args.peft_path, is_trainable=False)
    else:  # 训练模式初始化LoRA
        peft_config = LoraConfig(
            r=model_args.lora_rank,          # 低秩矩阵的秩
            lora_alpha=model_args.lora_alpha, # 缩放系数
            target_modules=model_args.trainable.split(","),  # 目标模块列表
            lora_dropout=model_args.lora_dropout,  # 防止过拟合
            task_type="SEQ_CLS",             # 序列分类任务
        )
        model = PeftModelForCLS(model, peft_config)  # 包装模型

    # 激活分类头参数（CLS Head）
    if training_args.do_train:
        for name, param in model.named_parameters():
            if "cls_head" in name:  # 只训练分类头
                param.requires_grad = True
    model.print_trainable_parameters()  # 打印可训练参数占比

    # 分词器加载 =========================================================
    print("********************** 加载分词器 **********************")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,  # 信任自定义模型代码
    )
    tokenizer.pad_token = tokenizer.unk_token  # 用UNK作为填充符
    tokenizer.padding_side = "right"  # 右侧填充（适合生成任务）

    # 数据加载与预处理 ====================================================
    data_files = {}  # 定义数据路径
    if data_args.train_file: data_files["train"] = data_args.train_file
    if data_args.validation_file: data_files["validation"] = data_args.validation_file
    if data_args.test_file: data_files["test"] = data_args.test_file

    # 加载JSON格式数据集
    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    print("原始数据集结构:", raw_datasets)

    # 动态选择数据集分割
    if training_args.do_train:
        target_dataset = raw_datasets["train"]
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        target_dataset = raw_datasets["eval"]
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        target_dataset = raw_datasets["test"]
        column_names = raw_datasets["test"].column_names

    # 数据预处理（格式转换）
    preprocess_func = llama_train_cls(
        data_args, model_args, tokenizer, ehr_tokenizer  # 输入参数
    )  # 返回预处理函数
    data_collator = LongestSequenceCollator(tokenizer)  # 动态填充至批次最大长度

    # 数据集映射预处理函数
    with training_args.main_process_first(desc="数据集预处理"):
        target_dataset = target_dataset.map(
            preprocess_func,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,  # 多进程处理
            remove_columns=column_names,  # 移除原始列
            desc="运行分词器处理数据",
        )
    target_dataset.set_format("torch")  # 转换为PyTorch张量

    # 训练器配置 ==========================================================
    trainer = MedRecTrainer(
        model=model,
        args=training_args,
        train_dataset=target_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,  # 自定义数据批处理
        compute_metrics=None,  # 不使用标准指标
        callbacks=([SavePeftModelCallback] if isinstance(model, PeftModel) else None),
    )

    # 训练流程 ============================================================
    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint  # 断点续训
        model.gradient_checkpointing_enable()  # 激活梯度检查点（节省显存）
        model.enable_input_require_grads()  # 确保输入张量需要梯度
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_state()  # 保存训练状态

    # 模型评估 ============================================================
    results = {}
    if training_args.do_predict:
        # 加载测试数据
        list_test_samples = []
        with open(data_args.test_file, "r", encoding="utf-8") as f:
            for line in f:
                list_test_samples.append(json.loads(line))

        # 执行预测
        start_time = time.time()
        with torch.no_grad():
            predict_results = trainer.predict(target_dataset, metric_key_prefix="predict")
        end_time = time.time()

        # 主进程保存结果
        if trainer.is_world_process_zero:
            predictions = predict_results.predictions
            hidden_states = predict_results.label_ids

            # 生成预测文件
            output_file = os.path.join(training_args.output_dir, "test_predictions.json")
            with open(output_file, "w", encoding="utf-8") as writer:
                for idx, p in enumerate(predictions):
                    sample = list_test_samples[idx]
                    sample["hidden_states"] = hidden_states[idx].astype(float).tolist()  # 隐藏层状态
                    sample["target"] = p.astype(float).tolist()  # 预测结果
                    writer.write(json.dumps(sample, ensure_ascii=False) + "\n")

            # 计算医疗推荐指标
            results = evaluate_jsonlines(output_file, ehr_tokenizer)

    return results

if __name__ == "__main__":
    train()  # 启动训练流程
