import os
import json
import random
import tqdm
import re
import argparse
import pandas as pd
from collections import OrderedDict
from templates.clf_task_template import template_1

import grpc
import llama_host_pb2
import llama_host_pb2_grpc

random.seed(42)

templates = {
    "template_1": template_1
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        default="data/gpt3_generations/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--num_instructions",
        type=int,
        help="if specified, only generate instance input for this many instructions",
    )
    parser.add_argument(
        "--template", 
        type=str, 
        default="template_1", 
        help="Which template to use. Currently only `template_1` is supported.",
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=4,
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

    with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl")) as fin:
        lines = fin.readlines()
        if args.num_instructions is not None:
            lines = lines[:args.num_instructions]

    output_path = os.path.join(args.batch_dir, f"is_clf_or_not_llama_{args.template}.jsonl")
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
    progress_bar = tqdm.tqdm(total=len(lines))
    with open(output_path, "w") as fout:
        for batch_idx in range(0, len(lines), args.request_batch_size):
            batch = [json.loads(line) for line in lines[batch_idx: batch_idx + args.request_batch_size]]
            if all(d["instruction"] in existing_requests for d in batch):
                for d in batch:
                    data = existing_requests[d["instruction"]]
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "is_classification"]
                        )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                # prefix = compose_prompt_prefix(human_written_tasks, batch[0]["instruction"], 8, 2)
                prefix = templates[args.template]
                prompts = [prefix + " " + d["instruction"].strip() + "\n" + "Is it classification?" for d in batch]
                results = make_requests(prompts, stub)

                stop_sequences = ["\n", "Task"]
                for i, _r in enumerate(results):
                    for _stop_seq in stop_sequences:
                        if _stop_seq in _r:
                            _r = _r[:_r.index(_stop_seq)]
                    results[i] = _r

                for i in range(len(batch)):
                    data = batch[i]
                    data["is_classification"] = results[i]

                    data = {
                        "instruction": data["instruction"],
                        "is_classification": data["is_classification"]
                    }
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "is_classification"]
                        )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            progress_bar.update(len(batch))