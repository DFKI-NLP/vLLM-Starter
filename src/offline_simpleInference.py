import argparse
from vllm import LLM, SamplingParams
from other.utils import load_model


def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Run text generation using an LLM.")
    parser.add_argument("--cache_dir", type=str, default="/path/to/your/models/", help="Path to the model cache directory.")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it", help="Name of the model to use.")
    parser.add_argument("--prompt", type=str, default="Write me a short story about a cat and a dog.", help="Prompt for text generation.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling parameter.")
    parser.add_argument("--max_tokens", type=int, default=3000, help="Maximum number of tokens to generate.")

    args = parser.parse_args()

    # Load the model
    print(f"Loading model '{args.model_name}' from '{args.cache_dir}'...")
    model = load_model(model_name=args.model_name, model_path=args.cache_dir)
    llm = LLM(model=model)

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

