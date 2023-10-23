import os
import json
import random
import re
import string
import tqdm
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
from rouge_score import rouge_scorer
import grpc
import llama_host_pb2
import llama_host_pb2_grpc

random.seed(42)

def encode_prompt(prompt_instructions, classification=False):
    """Encode multiple prompt instructions into a single string."""
    if classification:
        prompt = "Come up with a series of classification tasks. Try to specify the possible output labels when possible.\n"
    else:
        prompt = "Come up with a series of tasks:\n"
    for idx, instruction in enumerate(prompt_instructions):
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        prompt += f"{idx+1}. {instruction}\n"
    prompt += f"{len(prompt_instructions) + 1}."
    return prompt


def sample_machine_instructions(machine_instructions, similarities, n):
    """Sample n machine instructions from a list of machine instructions."""
    return random.sample(machine_instructions, min(n, len(machine_instructions)))


def find_word_in_string(w, s):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search(s)


def post_process_gpt3_response(response):
    raw_instructions = re.split(r"\n\d+\s?\. ", response)
    instructions = []
    for inst in raw_instructions:
        inst = re.sub(r"\s+", " ", inst).strip()
        inst = inst.strip().capitalize()
        if inst == "":
            continue
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        if any(find_word_in_string(word, inst) for word in ["image", "images", "graph", "graphs", "picture", "pictures", "file", "files", "map", "maps", "draw", "plot", "go to"]):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result. 
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append(inst)
    return instructions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        default="data/gpt3_generations/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        default="seed_tasks.jsonl",
        help="The path to the human written data.",
    )
    parser.add_argument(
        "--num_instructions_to_generate",
        type=int,
        default=20000,
        help="the target number of generated instructions.",
    )
    parser.add_argument(
        "--use_clf_seed_tasks_only",
        action="store_true",
        help="If specified, we will only use the classification seed tasks to prompt new instructions. This will lead to more classification instructions.",
    )
    parser.add_argument(
        "--num_prompt_instructions",
        type=int,
        default=8,
        help="The number of instructions to use in the prompt."
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=4,
        help="The number of requests to send to GPT3 at a time."
    )
    parser.add_argument(
        "--address",
        type=str,
        default="10.1.1.4:17425",
        help="the address that host llama grpc services."
    )
    return parser.parse_args()

def make_requests(prompt, stub):
    _request = llama_host_pb2.Prompt()
    _request.data.extend(prompt)
    return stub.complete(_request).data

if __name__ == "__main__":
    args = parse_args()
    seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
    if args.use_clf_seed_tasks_only:
        seed_tasks = [t for t in seed_tasks if t["is_classification"]]
    seed_instructions = [t["instruction"] for t in seed_tasks]
    print(f"Loaded {len(seed_instructions)} human-written seed instructions")
    
    os.makedirs(args.batch_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instructions = []
    if os.path.exists(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl")):
        with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl"), "r") as fin:
            for line in fin:
                instruction_info = json.loads(line)
                machine_instructions.append(instruction_info["instruction"])
                request_idx = instruction_info["request_idx"] + 1
        print(f"Loaded {len(machine_instructions)} machine-generated instructions")

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    
    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=args.num_instructions_to_generate)
    if machine_instructions:
        progress_bar.update(len(machine_instructions))

    channel = grpc.insecure_channel(args.address)
    stub = llama_host_pb2_grpc.llamahostStub(channel)
    
    with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl"), "a") as fout:
        while len(machine_instructions) < args.num_instructions_to_generate:
            try:
                batch_inputs = []
                for _ in range(args.request_batch_size):
                    # sample machine instructions from the pool
                    prompt_instructions = sample_machine_instructions(
                        machine_instructions, 
                        similarities=None,
                        n=2)
                    # sample human instructions from the pool
                    prompt_instructions += random.sample(seed_instructions, args.num_prompt_instructions - len(prompt_instructions))
                    random.shuffle(prompt_instructions)
                    prompt = encode_prompt(prompt_instructions, classification=args.use_clf_seed_tasks_only)
                    batch_inputs.append(prompt)
                results = make_requests(batch_inputs, stub)
                
                stop_sequences = ["\n\n", "\n16", "16.", "16 ."]
                for i, _r in enumerate(results):
                    for _stop_seq in stop_sequences:
                        if _stop_seq in _r:
                            _r = _r[:_r.index(_stop_seq)]
                    results[i] = _r
                
                instructions = []
                all_metadata = []
                for result in results:
                    new_instructions = post_process_gpt3_response(result)
                    instructions += new_instructions
                    all_metadata += [''] * len(new_instructions)

                for inst, metadata in zip(instructions, all_metadata):
                    with Pool(8) as p:
                        rouge_scores = p.map(partial(scorer.score, inst), seed_instructions + machine_instructions)
                    rouge_scores = [score["rougeL"].fmeasure for score in rouge_scores]
                    # rouge_scores = [scorer.score(inst, e_inst)["rougeL"].fmeasure for e_inst in human_instructions + machine_instructions]
                    if max(rouge_scores) > 0.7:
                        continue
                    all_instructions = seed_instructions + machine_instructions
                    most_similar_instructions = {
                            all_instructions[i] : rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
                        }
                    machine_instructions.append(inst)
                    fout.write(json.dumps({
                        "instruction": inst,
                        "most_similar": most_similar_instructions,
                        "avg_similarity_score": float(np.mean(rouge_scores)),
                        "metadata": metadata,
                        "request_idx": request_idx
                    }) + "\n")
                    progress_bar.update(1)
                request_idx += 1
            except KeyboardInterrupt:
                break