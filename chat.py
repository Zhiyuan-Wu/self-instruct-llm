import grpc
import llama_host_pb2
import llama_host_pb2_grpc

import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--address",
            type=str,
            default="10.1.1.4:17425",
            help="the address that host llama grpc services."
        )
    parser.add_argument(
            "--input",
            type=str,
            default="Can you help me to explain Generative AI to a five years old child.",
            help="The prompt."
        )
    args = parser.parse_args()

    channel = grpc.insecure_channel(args.address)
    stub = llama_host_pb2_grpc.llamahostStub(channel)

    question = args.input
    prompt = f"Task: You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Input: {question}. Output: "
    # prompt = f"Task: {question}. Output: "
    _request = llama_host_pb2.Prompt()
    _request.data.append(prompt)
    result = stub.complete(_request).data[0]
    stop_sequences = ["<|endoftext|>",]
    for _stop_seq in stop_sequences:
        if _stop_seq in result:
            result = result[:result.index(_stop_seq)]
    print(f"> {question}")
    print(f"> {result}")