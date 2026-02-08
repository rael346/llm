import heapq
import typing

import regex
from rich.progress import track

type Pair = tuple[int, int]
"""
(left, right)

byte pair
"""

type Chunk = tuple[list[int], int]
"""
(bytes, weight)

bytes: list of bytes of the chunk
weight: how many times this chunk appeared in the original text
"""

type PairItem = tuple[int, int, int, int]
"""
(count, neg_l, neg_r, id)

item in the priority queue for getting the max count pair
"""


def pair_str(pair: Pair):
    l, r = pair
    return f"Pair({chr(l) if l < 256 else l}, {chr(r) if r < 256 else r})"


def encode_with_special(
    merges: list[Pair],
    split_pattern: regex.Pattern[str],
    specials: list[str],
    text: str,
):
    special_pattern = "(" + "|".join(regex.escape(k) for k in specials) + ")"
    special_chunks = regex.split(special_pattern, text)
    ids = []
    for part in special_chunks:
        if part in specials:
            # this is a special token, encode it separately as a special case
            ids.append(256 + len(merges) + specials.index(part))
        else:
            # this is an ordinary sequence, encode it normally
            ids.extend(encode(merges, split_pattern, part))
    return ids


def encode(merges: list[Pair], split_pattern: regex.Pattern[str], text: str):
    text_chunks: list[str] = split_pattern.findall(text)
    chunks = [list(ch.encode("utf-8")) for ch in text_chunks]

    pair_count: dict[Pair, int] = {}
    pair_loc: dict[Pair, set[int]] = {}
    for i, chunk_bytes in enumerate(chunks):
        for pair in zip(chunk_bytes, chunk_bytes[1:]):
            if pair not in pair_loc:
                pair_loc[pair] = set()
            pair_count[pair] = pair_count.get(pair, 0) + 1
            pair_loc[pair].add(i)

    for i, merge_pair in enumerate(merges):
        if len(pair_count) == 0:
            break
        if merge_pair not in pair_count:
            continue
        del pair_count[merge_pair]

        for chunk_idx in pair_loc.pop(merge_pair):
            chunk_bytes = chunks[chunk_idx]
            deltas = merge_inplace(chunk_bytes, merge_pair, 256 + i)
            del deltas[merge_pair]

            for delta, count in deltas.items():
                if delta not in pair_loc:
                    pair_loc[delta] = set()
                pair_count[delta] = pair_count.get(delta, 0) + count

                if count > 0:
                    pair_loc[delta].add(chunk_idx)
                    continue

                if pair_count[delta] == 0:
                    del pair_count[delta]
                    del pair_loc[delta]
                    continue

                if count < 0 and not any(
                    delta[0] == l and delta[1] == r
                    for l, r in zip(chunk_bytes, chunk_bytes[1:])
                ):
                    pair_loc[delta].remove(chunk_idx)

    return [byte for ch in chunks for byte in ch]


def encode_inplace(merges: list[Pair], chunks: list[list[int]]):
    pair_count: dict[Pair, int] = {}
    pair_loc: dict[Pair, set[int]] = {}
    for i, chunk_bytes in enumerate(chunks):
        for pair in zip(chunk_bytes, chunk_bytes[1:]):
            if pair not in pair_loc:
                pair_loc[pair] = set()
            pair_count[pair] = pair_count.get(pair, 0) + 1
            pair_loc[pair].add(i)

    for i, merge_pair in enumerate(merges):
        if len(pair_count) == 0:
            break
        if merge_pair not in pair_count:
            continue
        del pair_count[merge_pair]

        for chunk_idx in pair_loc.pop(merge_pair):
            chunk_bytes = chunks[chunk_idx]
            deltas = merge_inplace(chunk_bytes, merge_pair, 256 + i)
            del deltas[merge_pair]

            for delta, count in deltas.items():
                if delta not in pair_loc:
                    pair_loc[delta] = set()
                pair_count[delta] = pair_count.get(delta, 0) + count

                if count > 0:
                    pair_loc[delta].add(chunk_idx)
                    continue

                if pair_count[delta] == 0:
                    del pair_count[delta]
                    del pair_loc[delta]
                    continue

                if count < 0 and not any(
                    delta[0] == l and delta[1] == r
                    for l, r in zip(chunk_bytes, chunk_bytes[1:])
                ):
                    pair_loc[delta].remove(chunk_idx)


def decode(vocab: list[bytes], specials: list[str], tokens: list[int]):
    decoded_bytes: list[bytes] = []
    for token in tokens:
        if token < len(vocab):
            decoded_bytes.append(vocab[token])
            continue
        decoded_bytes.append(specials[token - len(vocab)].encode("utf-8"))

    raw_bytes = b"".join(decoded_bytes)
    text = raw_bytes.decode("utf-8", errors="replace")
    return text


def train(chunk_count: dict[str, int], n_merges: int, verbose: bool):
    chunks: list[Chunk] = [
        (list(chunk_bytes.encode("utf-8")), weight)
        for chunk_bytes, weight in chunk_count.items()
    ]
    pair_count: dict[Pair, int] = {}
    pair_loc: dict[Pair, set[int]] = {}
    for i, (chunk_bytes, chunk_weight) in enumerate(
        track(chunks, "Calc pair Info", transient=not verbose)
    ):
        for pair in zip(chunk_bytes, chunk_bytes[1:]):
            if pair not in pair_loc:
                pair_loc[pair] = set()
            pair_count[pair] = pair_count.get(pair, 0) + chunk_weight
            pair_loc[pair].add(i)

    return get_merges_inplace(chunks, pair_count, pair_loc, n_merges, verbose)


def get_merges_inplace(
    chunks: list[Chunk],
    pair_count: dict[Pair, int],
    pair_loc: dict[Pair, set[int]],
    n_merges: int,
    verbose: bool = False,
):
    pair_queue = PairQueue(pair_count)
    merges: list[Pair] = []
    for i in track(range(n_merges), "Merging", transient=not verbose):
        max_pair = pair_queue.pop()
        merges.append(max_pair)
        del pair_count[max_pair]

        changed_pairs = set[Pair]()
        for chunk_idx in pair_loc.pop(max_pair):
            chunk_bytes, chunk_weight = chunks[chunk_idx]
            deltas = merge_inplace(chunk_bytes, max_pair, 256 + i)

            # max_pair has already been deleted in the pair_count
            del deltas[max_pair]
            changed_pairs.update(deltas.keys())

            for delta, count in deltas.items():
                if delta not in pair_loc:
                    pair_loc[delta] = set()
                pair_count[delta] = pair_count.get(delta, 0) + count * chunk_weight

                if count > 0:
                    pair_loc[delta].add(chunk_idx)
                    continue

                if pair_count[delta] == 0:
                    del pair_count[delta]
                    del pair_loc[delta]
                    continue

                if count < 0 and not any(
                    delta[0] == l and delta[1] == r
                    for l, r in zip(chunk_bytes, chunk_bytes[1:])
                ):
                    pair_loc[delta].remove(chunk_idx)

        for p in changed_pairs:
            count = pair_count[p] if p in pair_count else 0
            pair_queue.push(p, count)

    return merges


@typing.final
class PairQueue:
    def __init__(self, pair_count: dict[Pair, int]) -> None:
        self.heap: list[PairItem] = []
        self.pair_to_item: dict[Pair, PairItem] = {}

        for i, (pair, count) in enumerate(pair_count.items()):
            item: PairItem = (count, -pair[0], -pair[1], i)
            self.heap.append(item)
            self.pair_to_item[pair] = item

        self.counter = len(self.heap)
        self.old_items = set[int]()
        heapq.heapify_max(self.heap)

    def push(self, new_pair: Pair, count: int):
        if new_pair in self.pair_to_item:
            _, _, _, id = self.pair_to_item[new_pair]
            self.old_items.add(id)

        if count == 0:
            return

        new_item: PairItem = (count, -new_pair[0], -new_pair[1], self.counter)
        self.counter += 1
        self.pair_to_item[new_pair] = new_item
        heapq.heappush_max(self.heap, new_item)

    def pop(self) -> Pair:
        _, neg_l, neg_r, id = heapq.heappop_max(self.heap)
        while id in self.old_items:
            self.old_items.remove(id)
            _, neg_l, neg_r, id = heapq.heappop_max(self.heap)

        pair = (-neg_l, -neg_r)
        del self.pair_to_item[pair]
        return pair


def merge_inplace(
    chunk_bytes: list[int],
    max_pair: Pair,
    new_pair_idx: int,
):
    deltas: dict[Pair, int] = {}
    read_idx, write_idx = 0, 0
    while read_idx < len(chunk_bytes):
        if not (
            read_idx < len(chunk_bytes) - 1
            and chunk_bytes[read_idx] == max_pair[0]
            and chunk_bytes[read_idx + 1] == max_pair[1]
        ):
            chunk_bytes[write_idx] = chunk_bytes[read_idx]
            read_idx += 1
            write_idx += 1
            continue

        # adding max_pair here is technically not necessary
        # but mostly for correctness
        deltas[max_pair] = deltas.get(max_pair, 0) - 1
        chunk_bytes[write_idx] = new_pair_idx
        read_idx += 2
        write_idx += 1

        # Consider abcd -> aXd (bc -> X)
        # aXd
        #   _ write_idx
        if write_idx - 2 >= 0:
            # ab
            prev_old_pair = (chunk_bytes[write_idx - 2], max_pair[0])
            deltas[prev_old_pair] = deltas.get(prev_old_pair, 0) - 1

            # aX
            prev_new_pair = (chunk_bytes[write_idx - 2], new_pair_idx)
            deltas[prev_new_pair] = deltas.get(prev_new_pair, 0) + 1

        # aXd
        #   _ read_idx
        if read_idx < len(chunk_bytes):
            # cd
            next_old_pair = (max_pair[1], chunk_bytes[read_idx])
            deltas[next_old_pair] = deltas.get(next_old_pair, 0) - 1

            # Xd
            next_new_pair = (new_pair_idx, chunk_bytes[read_idx])
            deltas[next_new_pair] = deltas.get(next_new_pair, 0) + 1

    del chunk_bytes[write_idx:]

    # the only time when a count is 0 is when a non-existing pair
    # is created and then deleted immediately after
    # which is when two max_pair is next to each other
    # in this case there is no reason to add those pairs
    zero_pairs = [delta for delta, count in deltas.items() if count == 0]
    for del_pair in zero_pairs:
        del deltas[del_pair]

    return deltas


# def main():
#     GPT4_SPLIT_PATTERN = r"""
#         # shorten form of words like "will" ('ll), "have" ('ve), etc
#         '(?i:[sdmt]|ll|ve|re)
#         #
#         |[^\r\n\p{L}\p{N}]?+\p{L}+
#         # numbers between 0 - 999
#         |\p{N}{1,3}
#         |[ ]?[^\s\p{L}\p{N}]++[\r\n]*
#         |\s*[\r\n]
#         |\s+(?!\S)
#         |\s+
#         """
#     split_pattern = regex.compile(GPT4_SPLIT_PATTERN, regex.VERBOSE)
#
#     with open("raw_text.txt", "r", encoding="utf-8") as f:
#         text = f.read()
#
#     chunk_count: dict[str, int] = {}
#     for chunk in split_pattern.findall(text):
#         chunk_count[chunk] = 1 + chunk_count.get(chunk, 0)
#
#     chunks: list[Chunk] = [
#         (list(raw_text.encode("utf-8")), weight)
#         for raw_text, weight in chunk_count.items()
#     ]
#
#     merges, _ = get_merges(chunks, 1000)
#     vocab: list[bytes] = [bytes([i]) for i in range(256)]
#     for l, r in merges:
#         vocab.append(vocab[l] + vocab[r])
#
#     specials = [
#         "<|endoftext|>",
#         "<|fim_prefix|>",
#         "<|fim_middle|>",
#         "<|fim_suffix|>",
#         "<|endofprompt|>",
#     ]
#
#     test_text = "Hello've <|fim_prefix|> world123 <|fim_suffix|> how's are you!!!? <|endoftext|>"
#     encoded = encode_with_special(merges, split_pattern, specials, test_text)
#     print([chr(b) if b < 256 else b for b in encoded])
#     decoded = decode(vocab, specials, encoded)
#     print(decoded)
#
#
# if __name__ == "__main__":
#     main()
