import torch

from llm_py.gpt import GPTConfig
from llm_py.trainer import TrainConfig, Trainer


def main():
    with open("./dataset/input.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    vocab = sorted(list(set(raw_text)))
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}
    idx_to_char = {i: ch for i, ch in enumerate(vocab)}

    def encode(text: str):
        return [char_to_idx[c] for c in text]

    def decode(encoded: list[int]):
        return "".join(idx_to_char[i] for i in encoded)

    data = torch.tensor(encode(raw_text), dtype=torch.int64)

    test_model_config = GPTConfig(
        vocab_size=len(vocab),
        context_len=8,
        n_embd=32,
        n_blocks=3,
        n_head=4,
    )
    test_config = TrainConfig(
        batch_size=32,
        learning_rate=1e-3,
        train_iters=5000,
        eval_iterval=500,
        eval_iters=200,
        seed=1337,
        device="cpu",
    )

    scaled_model_config = GPTConfig(
        vocab_size=len(vocab),
        context_len=256,
        n_embd=384,
        n_blocks=6,
        n_head=6,
    )
    scaled_config = TrainConfig(
        batch_size=64,
        learning_rate=3e-4,
        train_iters=5000,
        eval_iterval=500,
        eval_iters=200,
        seed=1337,
        device="mps",
    )

    trainer = Trainer(scaled_config, scaled_model_config)
    trainer.fit(data)

    torch.save(trainer.model.state_dict(), "gpt.ckpt")

    # state_dict = torch.load("gpt.ckpt", weights_only=True)
    # model.load_state_dict(state_dict)

    eval_input = torch.zeros((1, 1), dtype=torch.int64, device=torch.device("mps"))
    print(decode(trainer.predict(eval_input, 500)[0].tolist()))


if __name__ == "__main__":
    main()
