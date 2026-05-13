from __future__ import annotations

import math
import os
import sys
from statistics import mean


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        print("CUDA unavailable")
        sys.exit(2)

    if os.environ.get("AUTONOMY_SMOKE_TINY") == "1":
        model_name = "hf-internal-testing/tiny-random-gpt2"
    else:
        model_name = "gpt2"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to("cuda")
    model.train()

    input_ids = torch.randint(
        0, model.config.vocab_size, (8, 128), device="cuda"
    )
    labels = input_ids.clone()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    losses: list[float] = []

    for _step in range(30):
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(math.isfinite(loss) for loss in losses), f"non-finite loss in {losses}"

    first_5_mean = mean(losses[:5])
    last_5_mean = mean(losses[-5:])
    assert last_5_mean < first_5_mean - 0.05, (
        f"loss did not decrease: first5={first_5_mean:.4f} last5={last_5_mean:.4f}"
    )

    print(
        f"gpt2-smoke pass first5={first_5_mean:.4f} last5={last_5_mean:.4f}"
    )


if __name__ == "__main__":
    main()
