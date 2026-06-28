import torch
import argparse
import numpy as np
from jamba_model_evolve import JambaLM, JambaLMConfig, JambaClassifier, get_pretrained_config

# ------------------------------------------------------------
def measure_latency(model, input_ids, num_warmup=50, num_runs=500):
    # Warm-up
    for _ in range(num_warmup):
        _ = model(input_ids)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(num_runs):
        start_event.record()
        _ = model(input_ids)
        end_event.record()
        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event))

    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # MiB
    timings = np.array(timings)
    mean = np.mean(timings)
    std = np.std(timings)
    median = np.median(timings)
    min_val = np.min(timings)
    max_val = np.max(timings)
    return mean, std, median, min_val, max_val, peak_mem

# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional path to a .pt checkpoint (if not given, model is randomly initialised).")
    parser.add_argument("--genotype", type=str, default=None,
                        help="Genotype as comma-separated list (e.g., '0,0,1,0,1'). Required if no checkpoint is given or if checkpoint lacks 'genotype'.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--num_runs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Determine genotype and optional state_dict
    genotype = None
    state_dict = None

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if 'genotype' in checkpoint:
            genotype = checkpoint['genotype']
            print("Genotype read from checkpoint.")
        elif args.genotype is not None:
            genotype = [int(x) for x in args.genotype.split(',')]
            print("Genotype provided manually (checkpoint lacked 'genotype').")
        else:
            print("ERROR: Checkpoint does not contain 'genotype' and --genotype was not given.")
            exit(1)

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint   # assume the whole object is a state_dict
    else:
        # No checkpoint: must have genotype from command line
        if args.genotype is None:
            print("ERROR: Either --checkpoint or --genotype must be provided.")
            exit(1)
        genotype = [int(x) for x in args.genotype.split(',')]
        print("No checkpoint given. Model will be randomly initialised.")

    print(f"Genotype: {genotype}")

    # Build model with Mini-Jamba configuration
    config = get_pretrained_config("TechxGenus/Mini-Jamba")
    base_lm = JambaLM(config, genotype).to(device)
    model = JambaClassifier(base_lm, args.num_classes).to(device)

    # Load weights if we have a state_dict
    if state_dict is not None:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: {len(missing)} missing keys (e.g., {missing[:3]})")
        if unexpected:
            print(f"Warning: {len(unexpected)} unexpected keys")
    else:
        print("Using fresh random weights (no checkpoint loaded).")

    model.eval()

    # Dummy input
    dummy_input = torch.randint(0, 20000, (args.batch_size, args.max_length), device=device)

    # Measure
    print(f"Measuring latency ({args.num_runs} runs)...")
    avg_ms, std_ms, median_ms, min_ms, max_ms, peak_mem = measure_latency(model, dummy_input)

    print(f"\nResults after {args.num_runs} runs:")
    print(f"  Mean:          {avg_ms:.2f} ms")
    print(f"  Median:        {median_ms:.2f} ms")
    print(f"  Std deviation:  {std_ms:.2f} ms")
    print(f"  Min:           {min_ms:.2f} ms")
    print(f"  Max:           {max_ms:.2f} ms")
    print(f"  95% interval:  [{avg_ms - 1.96*std_ms:.2f}, {avg_ms + 1.96*std_ms:.2f}] ms")
    print(f"  Peak memory:   {peak_mem:.2f} MiB")