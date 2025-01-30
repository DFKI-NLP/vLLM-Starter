import argparse
from vllm import LLM, SamplingParams

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Run text generation using an LLM.")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it", help="Name of the model to use.")
    parser.add_argument("--prompt", type=str, default="Write me a short story about a cat and a dog.", help="Prompt for text generation.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling parameter.")
    parser.add_argument("--max_tokens", type=int, default=3000, help="Maximum number of tokens to generate.")
    parser.add_argument("--trust-remote-code", type=bool, default=False, help="Allow remote code execution.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of tensor parallel partitions.")

    args = parser.parse_args()

    # Load the model
    print(f"Loading model '{args.model_name}'")
    model =args.model_name
    llm = LLM(model=model, trust_remote_code=args.trust_remote_code, tensor_parallel_size=args.tensor_parallel_size)

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )

    # Generate text
    print(f"Generating text for the prompt: {args.prompt!r}")
    outputs = llm.generate(args.prompt, sampling_params=sampling_params)

    # Print outputs
    for output in outputs:
        if hasattr(output, "outputs") and output.outputs:
            generated_text = output.outputs[0].text
            print(f"Prompt: {args.prompt!r}")
            print(f"Generated text: {generated_text!r}")
            print("---------------------------------------------------")
        else:
            print("No valid outputs generated.")


if __name__ == "__main__":
    main()

