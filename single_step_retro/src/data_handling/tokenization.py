import json
import re
from collections import Counter
from itertools import chain
from pathlib import Path
from typing import Iterable

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
MASK_TOKEN = "?"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

# pylint: disable=anomalous-backslash-in-string
REGEX = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
PATTERN = re.compile(REGEX)


def reverse_dict(d: dict[int | str, str | int]) -> dict[str | int, int | str]:
    """
    Reverses a dictionary: keys become values and values become keys.
    """
    return {v: k for k, v in d.items()}


def split_smiles_into_tokens(smi: str, check_reconstruction: bool = False) -> list[str]:
    """
    Splits a SMILES string into tokens according to the REGEX pattern proposed by Schwaller et al.
    """
    tokens = list(PATTERN.findall(smi))
    if check_reconstruction:
        assert smi == "".join(tokens)
    return tokens


class InplaceSMILESTokenizer:
    """
    This is a SMILES tokenizer for in-place SMILES modification.
    It has no start-of-sequence or end-of-sequence tokens.
    """

    def __init__(
        self,
        pad_token: str = PAD_TOKEN,
        bos_token: str = BOS_TOKEN,
        eos_token: str = EOS_TOKEN,
        unk_token: str = UNK_TOKEN,
        mask_token: str = MASK_TOKEN,
    ):
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = '?'#unk_token
        self.mask_token = mask_token

        self.pad_token_idx = 0
        self.bos_token_idx = 1
        self.eos_token_idx = 2
        self.unk_token_idx = 3
        self.mask_token_idx = 4

        self.service_tokens_idx = {
            self.pad_token_idx,
            self.bos_token_idx,
            self.eos_token_idx,
        }

        self.encoder_dict, self.decoder_dict = self._make_dictionaries()

    def __len__(self) -> int:
        return len(self.encoder_dict)

    def _make_dictionaries(self) -> tuple[dict[str, int], dict[int, str]]:
        encoder_dict = {
            self.pad_token: self.pad_token_idx,
            self.unk_token: self.unk_token_idx,
            self.mask_token: self.mask_token_idx,
            self.bos_token: self.bos_token_idx,
            self.eos_token: self.eos_token_idx,
        }
        decoder_dict = reverse_dict(encoder_dict)
        return encoder_dict, decoder_dict

    def train_tokenizer(self, train_data: Iterable[str]) -> None:
        """
        Goes through the training data, assembles the tokenizer dictionary (str to int)
        and saves it to train_data_path
        """
        tokenized_data = [
            split_smiles_into_tokens(line.strip(), check_reconstruction=True)
            for line in train_data
        ]
        token_counts = Counter(chain(*tokenized_data))

        for token, _ in token_counts.most_common():
            self.encoder_dict[token] = len(self.encoder_dict)

        self.decoder_dict = reverse_dict(self.encoder_dict)

    def save_vocab(self, voc_save_path: Path | str) -> None:
        """
        Saves the vocabulary dictionary which maps indices to strings.
        """
        p = Path(voc_save_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            json.dump(self.decoder_dict, f, sort_keys=True)

    def load_vocab(self, voc_load_path: Path | str) -> None:
        """
        Loads vocabulary dictionary which maps indices to strings
        """
        p = Path(voc_load_path).resolve()
        if not p.exists():
            raise FileNotFoundError
        with p.open() as f:
            self.decoder_dict = {int(k): v for k, v in json.load(f).items()}
            self.encoder_dict = reverse_dict(self.decoder_dict)

    def encode(self, seq: str) -> list[int]:
        """
        Turns a string into a list of token indices
        """
        unk = self.encoder_dict[self.unk_token]
        smi_tokens = split_smiles_into_tokens(seq, check_reconstruction=False)
        token_ids = [
            self.encoder_dict[i] if i in self.encoder_dict else unk for i in smi_tokens
        ]
        token_ids = [self.bos_token_idx, *token_ids, self.eos_token_idx]
        return token_ids

    def decode(self, tokens: Iterable[int], skip_service_tokens: bool = True) -> str:
        """
        Turns a list of token indices into a string.
        """
        if not skip_service_tokens:
            return "".join([self.decoder_dict[i] for i in tokens])
        result = []
        for i in tokens:
            if i == self.eos_token_idx:
                break
            if i not in self.service_tokens_idx:
                result.append(self.decoder_dict[i])
        return "".join(result)

    def decode_batch(
        self, tokens: list[Iterable[int]], skip_service_tokens: bool = True
    ) -> list[str]:
        """
        Turns a list of lists of token indices into a list of strings.
        """
        return [self.decode(i, skip_service_tokens=skip_service_tokens) for i in tokens]

    def decode_ctc_prediction(
        self,
        tokens: Iterable[int],
        blank_token: str = " ",
        skip_service_tokens: bool = True,
    ) -> str:
        """
        Decodes a CTC prediction.
        First, removes repeated tokens, then removes blank tokens.
        """
        if skip_service_tokens:
            string_tokens = [
                self.decoder_dict[i] for i in tokens if i not in self.service_tokens_idx
            ]
            result = []
            for i in tokens:
                if i == self.eos_token_idx:
                    break
                if i not in self.service_tokens_idx:
                    result.append(self.decoder_dict[i])
            string_tokens = result
        else:
            string_tokens = [self.decoder_dict[i] for i in tokens]

        # Remove repeated tokens
        string_tokens_filtered = [""]
        for token in string_tokens:
            if token != string_tokens_filtered[-1]:
                string_tokens_filtered.append(token)

        # Remove blank tokens
        string_tokens_filtered = [
            token for token in string_tokens_filtered if token != blank_token
        ]
        return "".join(string_tokens_filtered)

    def decode_ctc_prediction_batch(
        self,
        tokens: list[Iterable[int]],
        blank_token: str = " ",
        skip_service_tokens: bool = True,
    ) -> list[str]:
        """
        Decodes a batch of CTC predictions.
        """
        return [
            self.decode_ctc_prediction(token, blank_token, skip_service_tokens)
            for token in tokens
        ]
