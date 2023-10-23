import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from llama import Llama
import json
import argparse
import os
import tqdm
import loralib
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        default="data/gpt3_generations/finetuning/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="gpt3_finetuning_data_82436.jsonl",
        help="The path to the human written data.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="sft_lora_82K.pth",
        help="The path to lora model.",
    )
    parser.add_argument(
        "--epoch_num",
        type=int,
        default=5,
        help="th",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The number of requests to send to GPT3 at a time."
    )
    parser.add_argument(
        "--address",
        type=str,
        default="10.1.1.4:17425",
        help="the address that host llama grpc services."
    )
    return parser.parse_args()

class self_instruction_dataset(Dataset):
    def __init__(self, dataset_path):
        super().__init__()
        with open(dataset_path) as fin:
            lines = fin.readlines()
            self.instances = []
            for line in lines:
                data = json.loads(line)
                self.instances.append((data["prompt"], data["completion"]))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

if __name__ == "__main__":
    args = parse_args()
    dataset = self_instruction_dataset(os.path.join(args.batch_dir, args.input_file))
    dataset = DataLoader(dataset, batch_size=args.batch_size)
    model = Llama.build(
                        ckpt_dir="/home/zhiyuan/llama2/llama/llama-2-7b",
                        lora_path=None,
                        tokenizer_path="/home/zhiyuan/llama2/llama/tokenizer.model",
                        max_seq_len=2048,
                        max_batch_size=args.batch_size,
                        )

    total_para = 0
    trainable_para = 0
    for name, para in model.model.named_parameters():
        total_para += para.nelement()
        if para.requires_grad:
            trainable_para += para.nelement()
    print(f"total: {total_para}, tainable: {trainable_para}, ratio: {trainable_para/total_para}")
            
    optimizer = torch.optim.SGD(model.model.parameters(), lr=1e-4, )
    # optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-4, eps=1e-4)

    progress_bar = tqdm.tqdm(total=args.epoch_num * len(dataset))
    loss_ma = 0
    model.model.train()
    # torch.autograd.set_detect_anomaly(True)
    for e in range(args.epoch_num):
        for batch, (p, t) in enumerate(dataset):
            loss = model.compute_loss(p, t)
            loss_ma = 0.999*loss_ma + 0.001*loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss_ma)
            progress_bar.update(1)
            

        torch.save(loralib.lora_state_dict(model.model), args.ckpt_path)
            