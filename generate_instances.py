import os
import json
import random
import tqdm
import re
import argparse
import pandas as pd
from collections import OrderedDict
from templates.instance_gen_template import output_first_template_for_clf, input_first_template_for_gen

import grpc
import llama_host_pb2
import llama_host_pb2_grpc

random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        default="data/gpt3_generations/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="machine_generated_instructions.jsonl"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="machine_generated_instances.jsonl",
    )
    parser.add_argument(
        "--num_instructions",
        type=int,
        help="if specified, only generate instance input for this many instructions",
    )
    parser.add_argument(
        "--max_instances_to_generate",
        type=int,
        default=5,
        help="The max number of instances to generate for each instruction.",
    )
    parser.add_argument(
        "--generation_tasks_only",
        action="store_true",
        help="If specified, only do for generation tasks.",
    )
    parser.add_argument(
        "--classification_tasks_only",
        action="store_true",
        help="If specified, only do for classification tasks.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="llama",
        help="The engine to use."
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=2,
        help="The number of requests to send in a batch."
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

if __name__ == '__main__':
    args = parse_args()

    with open(os.path.join(args.batch_dir, args.input_file)) as fin:
        lines = fin.readlines()
        if args.num_instructions is not None:
            lines = lines[:args.num_instructions]
        tasks = []
        for line in lines:
            data = json.loads(line)
            if "metadata" in data:
                data["instruction_metadata"] = data["metadata"]
                del data["metadata"]
            tasks.append(data)

    task_clf_types = {}
    with open(os.path.join(args.batch_dir, "is_clf_or_not_llama_template_1.jsonl")) as fin:
        for line in fin:
            data = json.loads(line)
            task_clf_types[data["instruction"]] = data["is_classification"].strip() in ["Yes", "yes", "YES"]

    if args.classification_tasks_only:
        tasks = [task for task in tasks if task_clf_types[task["instruction"]]]
    
    if args.generation_tasks_only:
        tasks = [task for task in tasks if not task_clf_types[task["instruction"]]]

    output_path = os.path.join(args.batch_dir, args.output_file)
    existing_requests = {}
    if os.path.exists(output_path):
        with open(output_path) as fin:
            for line in tqdm.tqdm(fin):
                try:
                    data = json.loads(line)
                    existing_requests[data["instruction"]] = data
                except:
                    pass
        print(f"Loaded {len(existing_requests)} existing requests")

    channel = grpc.insecure_channel(args.address)
    stub = llama_host_pb2_grpc.llamahostStub(channel)
    
    progress_bar = tqdm.tqdm(total=len(tasks))
    with open(output_path, "w") as fout:
        for batch_idx in range(0, len(tasks), args.request_batch_size):
            batch = tasks[batch_idx: batch_idx + args.request_batch_size]
            if all(d["instruction"] in existing_requests for d in batch):
                for d in batch:
                    data = existing_requests[d["instruction"]]
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "raw_instances", "instance_metadata", "instruction_metadata", 
                            "most_similar", "avg_similarity_score"]
                        )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                prompts = []
                for task in batch:
                    if task_clf_types[task["instruction"]]:
                        prompt = output_first_template_for_clf + " " + task["instruction"].strip() + "\n"
                        prompts.append(prompt)
                    else:
                        prompt = input_first_template_for_gen + " " + task["instruction"].strip() + "\n"
                        prompts.append(prompt)
                results = make_requests(prompts, stub)
                
                stop_sequences=[f"Example {args.max_instances_to_generate + 1}", "Task:"]
                for i, _r in enumerate(results):
                    for _stop_seq in stop_sequences:
                        if _stop_seq in _r:
                            _r = _r[:_r.index(_stop_seq)]
                    results[i] = _r

                for i in range(len(batch)):
                    data = batch[i]
                    data["instance_metadata"] = ''
                    data["raw_instances"] = results[i]
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "raw_instances", "instance_metadata", "instruction_metadata", 
                            "most_similar", "avg_similarity_score"]
                        )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            progress_bar.update(len(batch))