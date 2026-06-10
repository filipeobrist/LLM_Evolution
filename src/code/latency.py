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

    timings = []
    for _ in range(num_runs):
        start_event.record()
        _ = model(input_ids)
        end_event.record()
        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event))

    timings = np.array(timings)
    mean = np.mean(timings)
    std = np.std(timings)
    median = np.median(timings)
    min_val = np.min(timings)
    max_val = np.max(timings)
    return mean, std, median, min_val, max_val

# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--num_runs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--genotype", type=str, default=None,
                        help="Genótipo manual (ex: '0,0,1,0,1'). Só necessário se o checkpoint não tiver 'genotype'.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # Carregar checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Determinar genótipo
    if 'genotype' in checkpoint:
        genotype = checkpoint['genotype']
        print("Genótipo lido do checkpoint.")
    elif args.genotype is not None:
        genotype = [int(x) for x in args.genotype.split(',')]
        print("Genótipo fornecido manualmente.")
    else:
        print("ERRO: O checkpoint não contém 'genotype' e não foi fornecido --genotype.")
        exit(1)

    print(f"Genótipo: {genotype}")

    # Usar a configuração exata do Mini‑Jamba (a mesma com que o modelo foi treinado)
    config = get_pretrained_config("TechxGenus/Mini-Jamba")
    base_lm = JambaLM(config, genotype).to(device)
    model = JambaClassifier(base_lm, args.num_classes).to(device)

    # Carregar pesos (estado pode estar na chave 'state_dict' ou diretamente no dicionário)
    if 'state_dict' in checkpoint:
        state = checkpoint['state_dict']
    else:
        state = checkpoint

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Aviso: {len(missing)} chaves em falta (ex: {missing[:3]})")
    if unexpected:
        print(f"Aviso: {len(unexpected)} chaves inesperadas")

    model.eval()

    # Input aleatório
    dummy_input = torch.randint(0, 20000, (args.batch_size, args.max_length), device=device)

    # Medir
    print(f"A medir latência ({args.num_runs} execuções)...")
    avg_ms, std_ms, median_ms, min_ms, max_ms = measure_latency(model, dummy_input)

    print(f"\nResultado após {args.num_runs} execuções:")
    print(f"  Média:        {avg_ms:.2f} ms")
    print(f"  Mediana:      {median_ms:.2f} ms")
    print(f"  Desvio padrão: {std_ms:.2f} ms")
    print(f"  Mínimo:       {min_ms:.2f} ms")
    print(f"  Máximo:       {max_ms:.2f} ms")
    print(f"  Intervalo 95%: [{avg_ms - 1.96*std_ms:.2f}, {avg_ms + 1.96*std_ms:.2f}] ms")


# A latência foi medida com o modelo em modo de avaliação, 
# utilizando eventos CUDA para precisão. Foram realizadas 
# 50 iterações de aquecimento seguidas de 500 medições. 
# A média e o desvio padrão dessas medições são reportados. 
# As medições foram efetuadas com a GPU dedicada exclusivamente 
# ao processo, garantindo condições estáveis.