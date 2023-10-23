# Self-Instruction for LLM

This repo is an experimental code that use self-generated instruction data from Llama to fine-tune a LoRA patch over it, in order to derive an instruction-aligned chat model.

The code is gathered from multiple sources (see below), and is simplified for better code readability (e.g., remove model parallesim and unfold implementation details). Therefore the code is not optimized for performance but aims to help learning main concepts about llm. The code can be run (and tested) on a single 24GB-RAM RTX 3090 card (for 7B model, f16 precision, 4k-token context).

## Usage
Some main steps are as follows. Specific parameters usage can be found correspondingly in related scripts.

1. Download Llama-2 pretrained weights.
```bash
bash download.sh
```

2. Start a gRPC service to process queries online. Server parameters should be adjusted later for different scripts.
```bash
python -m grpc_tools.protoc -I . --python_out=. --pyi_out=. --grpc_python_out=. llama_host.proto
python llama_host.py
```

3. Generate instructions (tasks) using the few-shot ability of llm. Server should be set to `--max_seq_len 4096 --max_gen_len 500 --max_batch_size 4`
```bash
python boostrap_instruction.py
```

4. Classify generated instructions using the few-shot ability of llm. Server should be set to `--max_seq_len 4096 --max_gen_len 3 --max_batch_size 4`
```bash
python identify_clf_or_not.py
```

5. Generate instances for generated instructions using the few-shot ability of llm. Server should be set to `--max_seq_len 8192 --max_gen_len 500 --max_batch_size 2`
```bash
python generate_instances.py
```

6. Fine-tune the pre-trained model by LoRA PEFT, to let model follow input instructions beyond simple completion.
```bash
python prepare_for_fintuning.py
python fine_tune.py
```

7. Define proper prompts and chat with trained model. Server should be set with `--lora_path sft_lora.pth`
```bash
python chat.py
```

## Code Source
The code is constructed based on following repos:
- [facebookresearch/llama](https://github.com/facebookresearch/llama)
- [yizhongw/self-instruct](https://github.com/yizhongw/self-instruct)
- [microsoft/LoRA](https://github.com/microsoft/LoRA)

Zhiyuan