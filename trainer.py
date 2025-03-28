from trl import GRPOConfig, GRPOTrainer
from typing import Any, Callable, Optional, Sized, Union, List, Tuple

import os
import textwrap
import warnings
from collections import defaultdict
from unittest.mock import patch

import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available


    
class GRPPOTrainer():
    def __init__(
        self,
        model_parts: List[nn.Module],
        reward_funcs: list[Callable[...,Any]],
        optimizers: Tuple[torch.optim.Optimizer, float],
        DataLoader=None,
        args: GRPOConfig=None,
        compare_functions: list[Callable[...,Any]]=None,
        train_dataset=None,
        test_dataset=None,
    ):
        self.model          = nn.Sequential(*model_parts)
        self.model_parts    = model_parts
        self.forward_parts  = [nn.Sequential(*(model_parts[:i])) for i in range(1,len(model_parts)+1)]
        self.blocked_parts  = [nn.Sequential(*(model_parts[i:])) for i in range(1,len(model_parts)+1)]
        self.rewards_funcs  = reward_funcs
        self.optimizers     = [optimizers[0](part.parameters(),lr=optimizers[1]) for part in model_parts]
        self.DataLoader     = DataLoader
        if args is not None and args.reward_weights is None:
            args.reward_weights = torch.full((len(reward_funcs),), 1.0).to(args.device)
            
        self.args           = args
        print(self.args.reward_weights)
        for part in model_parts:
            part.to(self.args.device) 
        if compare_functions is None:
            self.compare_functions=[None]*len(model_parts)
        else:
            self.compare_functions=compare_functions
        self.default_compare= lambda a,b:torch.nn.functional.cosine_similarity(a, b, dim=-1)
        self.train_dataset  = train_dataset
        self.test_dataset   = test_dataset

    def get_loss(self, 
        inputs, 
        forward_model, 
        blocked_model, 
        reward_funcs, 
        compare_function=None,
        reward_weights=None, 
        reward_targets=None, 
        num_generations=None):
        if compare_function is None:
            compare_function = self.default_compare
        if num_generations is None:
            num_generations = self.num_generations
        if reward_targets is None:
            reward_targets = [None] * len(reward_funcs)
        
        # inputs 是一个 batch 的张量，形状为 [batch_size, input_dim]
        device = inputs.device


        """repeated_inputs = inputs.unsqueeze(0).repeat(num_generations, 1, 1)  # 形状 [num_generations * batch_size, input_dim]
        with torch.no_grad():
            targets = forward_model(repeated_inputs)
            outputs = blocked_model(targets)"""

        # 将 inputs 重复 num_generations 次，从而生成这么多样本
        targets_list = []
        outputs_list = []
        with torch.no_grad():
            for _ in range(num_generations):
                targets = forward_model(inputs)  # 每次只传入原始 inputs
                outputs = blocked_model(targets)
                targets_list.append(targets)
                outputs_list.append(outputs)
            outputs = torch.stack(outputs_list, dim=0)
            targets = torch.stack(targets_list, dim=0)
        # 计算原始 rewards
        rewards_per_func = []
        for f,target in zip(reward_funcs,reward_targets):
            if target is not None:
                t=target.unsqueeze(0).repeat(num_generations, 1, 1)
                r = f(outputs, t)
            else:
                r = f(outputs)
            # 确保奖励形状为 [num_gens * batch_size,]
            if r.ndim > 1:
                r = r.squeeze(-1)
            rewards_per_func.append(r)
        
        # 加权求和
        rewards_per_func = torch.stack(rewards_per_func, dim=-1)  # [num_gens, batch, n_func]
        total_rewards = (rewards_per_func * reward_weights).sum(dim=-1)  # [num_gens, batch]
        
        # 分组归一化计算优势
        mean_rewards = total_rewards.mean(dim=0, keepdim=True)  # [1, batch]
        std_rewards = total_rewards.std(dim=0, keepdim=True)    # [1, batch]
        advantages = (total_rewards - mean_rewards) / (std_rewards + 1e-4)
        
        # 重新前向传播获取新输出
        new_outputs = forward_model(inputs)
        
        # 扩展维度用于批量比较
        new_outputs_expanded = new_outputs.unsqueeze(0).expand(num_generations, self.args.batch_size, -1)
        # 计算相似度
        similarities = compare_function(new_outputs_expanded, targets)
        # 用归一化后的优势加权
        loss = -(similarities * advantages).mean()
        
        return loss, targets, advantages

    def train_step(self, inputs, target=None):
        total_loss = 0.0
        inputs.to(self.args.device)
        # Get the total number of parts
        num_parts = len(self.model_parts)
        
        # Traverse each model part with its corresponding components
        for model_part, forward_part, blocked_part, optimizer, compare_function in zip(
            self.model_parts, self.forward_parts, self.blocked_parts, self.optimizers, self.compare_functions
        ):
            # Set current module to training mode, others to eval mode
            model_part.train()
            for p in self.model_parts:
                if p is not model_part:
                    p.eval()
            
            # Calculate loss differently based on whether it's the last part
            loss, _, _ = self.get_loss(
                inputs=inputs,
                forward_model=forward_part,
                blocked_model=blocked_part,
                reward_funcs=self.rewards_funcs,
                reward_weights=self.args.reward_weights,
                reward_targets=[target],  # Changed from target to None for non-last parts
                num_generations=self.args.num_generations
            )
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            if self.args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model_part.parameters(), self.args.max_grad_norm)
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.model_parts)
            
    def train(self):
        dataloader = self.DataLoader(self.train_dataset, batch_size=self.args.batch_size)
        for epoch in range(self.args.epochs):
            epoch_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(self.args.device)
                targets = targets.to(self.args.device) if targets is not None else None
                
                loss = self.train_step(inputs, targets)
                epoch_loss += loss
                
                if batch_idx % self.args.log_interval == 0:
                    print(f"Epoch {epoch} Batch {batch_idx} Loss: {loss:.4f}")
            
            print(f"Epoch {epoch} Avg Loss: {epoch_loss/len(dataloader):.4f}")
        return self.model

    def prediction(self):
        dataloader = self.DataLoader(self.train_dataset, batch_size=1)
        return model(dataloader[0])