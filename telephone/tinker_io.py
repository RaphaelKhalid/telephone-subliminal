"""Thin wrapper over the Tinker SDK.

Two notes, both learned the hard way from reading the shipped package rather
than the docs:

  * `SampleResponse` has `.sequences`, not `.samples`, and sequences have no
    `.text` -- decode the token ids yourself.
  * `ForwardBackwardOutput` has no `.loss`. Metrics live in `.metrics`.

Both of those appear incorrectly in the SDK's own docstrings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import re

import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

from . import config

CHUNK = 64  # how many sampling requests to have in flight at once

_THINK = re.compile(r"^.*?</think>", re.DOTALL)


_SPECIAL = re.compile(r"<\|[^>]*\|>")


def strip_thinking(text: str) -> str:
    """Belt and braces. The renderer should prevent this; if a thinking block
    appears anyway, drop it rather than letting it poison the training data."""
    return _THINK.sub("", text).replace("<think>", "").strip()


def decode_clean(rig: "Rig", tokens) -> str:
    """Decode a sampled sequence into text fit for the parser.

    `skip_special_tokens` handles the usual case. The regex is a backstop: the
    stop token can arrive as ordinary text rather than a special id, and a
    trailing "<|im_end|>" welded onto the last number is enough to make the
    whole sequence unparseable and get it thrown away by the filter.
    """
    try:
        text = rig.tokenizer.decode(tokens, skip_special_tokens=True)
    except TypeError:
        text = rig.tokenizer.decode(tokens)
    text = _SPECIAL.sub("", text)
    for s in (rig.stop or []):
        if isinstance(s, str):
            text = text.replace(s, "")
    return strip_thinking(text)


@dataclass
class Rig:
    service: "tinker.ServiceClient"
    tokenizer: object
    renderer: object
    stop: object

    @classmethod
    def build(cls, base_model: str = config.BASE_MODEL) -> "Rig":
        service = tinker.ServiceClient()
        tokenizer = get_tokenizer(base_model)
        renderer_name = getattr(config, "RENDERER_NAME", None) or \
            model_info.get_recommended_renderer_name(base_model)
        renderer = renderers.get_renderer(renderer_name, tokenizer)
        print(f"renderer: {renderer_name}")
        return cls(service, tokenizer, renderer, renderer.get_stop_sequences())

    def sampling_client(self, model_path: str | None = None):
        if model_path:
            return self.service.create_sampling_client(model_path=model_path)
        return self.service.create_sampling_client(base_model=config.BASE_MODEL)


def sample_many(
    rig: Rig,
    client,
    user_prompts: list[str],
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    num_samples: int = 1,
) -> tuple[list[list[str]], int]:
    """Sample from `client` for each prompt.

    Returns (completions_per_prompt, total_tokens_billed). Token count includes
    the prompt, since prefill is billed too.
    """
    params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
        stop=rig.stop,
    )

    def to_input(p: str):
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": p})
        return rig.renderer.build_generation_prompt(msgs)

    out: list[list[str]] = []
    total_tokens = 0

    for start in range(0, len(user_prompts), CHUNK):
        chunk = user_prompts[start: start + CHUNK]
        futures = []
        for p in chunk:
            mi = to_input(p)
            total_tokens += mi.length() * num_samples
            futures.append(
                client.sample(
                    prompt=mi, num_samples=num_samples, sampling_params=params
                )
            )
        for fut in futures:
            resp = fut.result()
            texts = []
            for seq in resp.sequences:          # .sequences, not .samples
                total_tokens += len(seq.tokens)
                texts.append(decode_clean(rig, seq.tokens))
            out.append(texts)
        done = min(start + CHUNK, len(user_prompts))
        print(f"      sampled {done}/{len(user_prompts)}", flush=True)

    return out, total_tokens


def train_lora(
    rig: Rig,
    pairs: list[tuple[str, str]],
    tag: str,
    epochs: int = config.N_EPOCHS,
) -> tuple[str, int]:
    """Fine-tune a fresh LoRA on (user_prompt, assistant_completion) pairs.

    Every generation starts from the base model, not from its parent. The
    parent's influence travels only through the data -- which is the claim
    being tested.

    Returns (checkpoint_path, train_tokens_billed).
    """
    client = rig.service.create_lora_training_client(
        base_model=config.BASE_MODEL,
        rank=config.LORA_RANK,          # the kwarg is `rank`
        user_metadata={"tag": tag},
    )

    data = []
    train_tokens = 0
    for prompt, completion in pairs:
        conv = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
        datum = conversation_to_datum(
            conv,
            rig.renderer,
            config.MAX_SEQ_LENGTH,
            train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            reduction="mean",
        )
        train_tokens += datum.model_input.length()
        data.append(datum)

    train_tokens *= epochs

    n_batches = len(data) // config.BATCH_SIZE
    total_steps = n_batches * epochs
    step = 0
    t0 = time.time()

    for epoch in range(epochs):
        for b in range(n_batches):
            batch = data[b * config.BATCH_SIZE: (b + 1) * config.BATCH_SIZE]
            lr = config.LEARNING_RATE * max(0.05, 1.0 - step / max(total_steps, 1))
            adam = types.AdamParams(
                learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8
            )
            # Submit both before awaiting either: one clock cycle, not three.
            fb = client.forward_backward(batch, loss_fn="cross_entropy")
            os_ = client.optim_step(adam)
            fb_result = fb.result()
            os_.result()
            step += 1
            if step % 10 == 0 or step == total_steps:
                loss = fb_result.metrics.get("loss:sum", "?")
                print(
                    f"      step {step}/{total_steps}  lr={lr:.2e}  "
                    f"loss={loss}  {time.time()-t0:.0f}s",
                    flush=True,
                )

    path = client.save_weights_for_sampler(
        name=f"{tag}-final", ttl_seconds=None
    ).result().path
    print(f"      checkpoint: {path}", flush=True)
    return path, train_tokens
