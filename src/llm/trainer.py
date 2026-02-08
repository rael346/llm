from dataclasses import dataclass
from typing import Literal, final

import torch
import torch.nn.functional as F
from rich.progress import track

from llm.gpt import GPT, GPTConfig


@dataclass
class TrainConfig:
    batch_size: int
    learning_rate: float
    train_iters: int
    eval_iterval: int
    eval_iters: int
    seed: int
    device: Literal["cpu", "mps", "cuda"]


@final
class Trainer:
    def __init__(self, config: TrainConfig, model_config: GPTConfig) -> None:
        self.rng = torch.manual_seed(config.seed)
        self.c = config
        self.model_c = model_config
        self.device = torch.device(config.device)
        self.model = GPT(model_config).to(self.device)
        if self.c.device != "mps":
            self.model.compile()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate
        )

    def gen_batch(self, data: torch.Tensor):
        start_idx = torch.randint(
            0,
            data.shape[0] - self.model_c.context_len,
            (self.c.batch_size,),
            generator=self.rng,
        )
        train = torch.stack([data[i : i + self.model_c.context_len] for i in start_idx])
        labels = torch.stack(
            [data[i + 1 : i + self.model_c.context_len + 1] for i in start_idx]
        )
        return train.to(self.device), labels.to(self.device)

    def fit(self, data: torch.Tensor):
        train_len = int(0.9 * data.shape[0])
        train_data, eval_data = data[:train_len], data[train_len:]

        total_train_loss = torch.zeros(1, device=self.device)
        for iter in track(range(self.c.train_iters), "Train"):
            if iter % self.c.eval_iterval == 0 or iter == self.c.train_iters - 1:
                self.model.eval()
                avg_eval = self.eval(eval_data)
                avg_train = total_train_loss.item() / self.c.eval_iterval
                print(
                    (f"Iter {iter:04d} | Train: {avg_train:.4f} | Eval: {avg_eval:.4f}")
                )
                total_train_loss = torch.zeros(1, device=self.device)
                self.model.train()

            train, labels = self.gen_batch(train_data)
            pred = self.model(train).view((-1, self.model_c.vocab_size))
            actual = labels.view(-1)

            loss = F.cross_entropy(pred, actual)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total_train_loss += loss

    def eval(self, data: torch.Tensor):
        total_loss = torch.zeros(1, device=self.device)
        with torch.no_grad():
            for iter in range(self.c.eval_iters):
                batch, labels = self.gen_batch(data)
                pred = self.model(batch).view((-1, self.model_c.vocab_size))
                actual = labels.view(-1)

                loss = F.cross_entropy(pred, actual)
                total_loss += loss
        return total_loss.item() / self.c.eval_iters

    def predict(self, input: torch.Tensor, n_next_tokens: int):
        self.model.eval()
        output = input.clone()
        with torch.no_grad():
            for _ in range(n_next_tokens):
                output_truncated = output[:, -self.model_c.context_len :]
                # get the predictions
                logits = self.model(output_truncated)
                # focus only on the last time step
                logits = logits[:, -1, :]  # becomes (B, C)
                # apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1)  # (B, C)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
                # append sampled index to the running sequence
                output = torch.cat((output, idx_next), dim=1)  # (B, T+1)
        return output
