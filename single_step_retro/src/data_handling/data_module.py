from itertools import chain
from pathlib import Path

import torch
from pytorch_lightning import LightningDataModule
from torch.nn.functional import pad
from torch.utils.data import DataLoader, Dataset

from single_step_retro.src.data_handling.batching import TokenSampler
from single_step_retro.src.data_handling.tokenization import InplaceSMILESTokenizer
from single_step_retro.src.utils.rotation import get_vocab_rotation_target


class Smiles2SmilesInPlaceDataset(Dataset):
    def __init__(
        self,
        src_path: Path | str,
        tgt_path: Path | str,
        max_size: int,
        tokenizer: InplaceSMILESTokenizer,
    ):
        self.tokenizer = tokenizer
        self.max_size = max_size
        with open(src_path) as fs, open(tgt_path) as ft:
            self.source: list[str] = [s.strip() for s in fs.readlines()][:]
            self.target: list[str] = [s.strip() for s in ft.readlines()][:]
        assert len(self.source) == len(self.target), (
            f"The source and target data at {src_path} and {tgt_path} have different lenghts"
        )
        self.source_tokens = [self.tokenizer.encode(s) for s in self.source]
        self.target_tokens = [self.tokenizer.encode(s) for s in self.target]

        self.source_lengths = [len(i) for i in self.source_tokens]
        self.target_lengths = [len(i) for i in self.target_tokens]

    def __len__(self):
        return len(self.source_tokens)

    def __getitem__(self, item):
        src = torch.tensor(self.source_tokens[item]).long()
        tgt = torch.tensor(self.target_tokens[item]).long()

        src = pad(
            src,
            (0, self.max_size - src.size(-1)),
            "constant",
            self.tokenizer.pad_token_idx,
        )
        tgt = pad(
            tgt,
            (0, self.max_size - tgt.size(-1)),
            "constant",
            self.tokenizer.pad_token_idx,
        )

        return src, tgt


class Smiles2SmilesInPlaceDM(LightningDataModule):
    """
    A Lightning Data Module specific to non-autoregressive SMILES-to-SMILES conversion.

    Args:
        :param data_dir: Path to the directory containing the data files in the format {src,tgt}-{train,val,test}.txt.
        If this format is not used, the paths to the source and target training, validation, and test data files can be provided explicitly.
        :param src_train_path: Optional path to the source training data file.
        :param tgt_train_path: Optional path to the target training data file.
        :param src_val_path: Optional path to the source validation data file.
        :param tgt_val_path: Optional path to the target validation data file.
        :param src_test_path: Optional path to the source test data file.
        :param tgt_test_path: Optional path to the target test data file.
        :param vocab_path: Path to the vocabulary file.
        :param max_size: Maximum size of the input/output sequences.
        The sequences in batches will are padded to have this length.
        :param rotation_target: If true, the model will predict shifts in tokens between source and target
        instead of predicting the target tokens directly.
        :param batch_size: Number of samples per batch.
        :param tokens_in_batch: Maximum number of tokens in a batch. If provided, batching will try to minimize
        the proportion of pad token in batches by having sequences of similar length in evety batch.
        :param num_workers: Number of subprocesses to use for data loading.
        :param persistent_workers: Whether to maintain the workers between iterations.
        :param pin_memory: Whether to pin memory in data loader.
        :param shuffle_train: Whether to shuffle the training data.
    """

    def __init__(
        self,
        data_dir: str | None = None,  # Data location arguments
        src_train_path: str | None = None,
        tgt_train_path: str | None = None,
        src_val_path: str | None = None,
        tgt_val_path: str | None = None,
        src_test_path: str | None = None,
        tgt_test_path: str | None = None,
        vocab_path: str | None = None,
        max_size: int = 200,
        rotation_target: bool = False,
        batch_size: int = 1,  # Batching arguments
        tokens_in_batch: int | None = None,
        num_workers: int = 0,
        persistent_workers=False,
        pin_memory: bool = False,
        shuffle_train: bool = False,
    ):
        super().__init__()

        self.data_dir = Path(data_dir).resolve()
        self.vocab_path = vocab_path
        self.max_size = max_size
        self.rotation_target = rotation_target
        self.src_train_path = src_train_path or self.data_dir / "src-train.txt"
        self.tgt_train_path = tgt_train_path or self.data_dir / "tgt-train.txt"
        self.src_val_path = src_val_path or self.data_dir / "src-val.txt"
        self.tgt_val_path = tgt_val_path or self.data_dir / "tgt-val.txt"
        self.src_test_path = src_test_path or self.data_dir / "src-test.txt"
        self.tgt_test_path = tgt_test_path or self.data_dir / "tgt-test.txt"

        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.tokens_in_batch = tokens_in_batch

        self.tokenizer = self.create_tokenizer()

    def create_tokenizer(self):
        if self.vocab_path is None:
            self.vocab_path = self.data_dir / "vocabs" / "vocab.json"
        else:
            self.vocab_path = Path(self.vocab_path).resolve()

        tokenizer = InplaceSMILESTokenizer()
        try:
            tokenizer.load_vocab(self.vocab_path)
            print(f"Loaded tokenizer vocabulary from {self.vocab_path}")
        except FileNotFoundError:
            print("Training tokenizer...")
            with open(self.src_train_path) as f, open(self.tgt_train_path) as g:
                tokenizer.train_tokenizer(chain(f, g))
            tokenizer.save_vocab(self.vocab_path)
            print(f"Saved tokenizer vocab to: {self.vocab_path}")
        finally:
            return tokenizer

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train = Smiles2SmilesInPlaceDataset(
                self.src_train_path, self.tgt_train_path, self.max_size, self.tokenizer
            )
            self.val = Smiles2SmilesInPlaceDataset(
                self.src_val_path, self.tgt_val_path, self.max_size, self.tokenizer
            )

        if stage == "validate":
            self.val = Smiles2SmilesInPlaceDataset(
                self.src_val_path, self.tgt_val_path, self.max_size, self.tokenizer
            )

        if stage == "test" or stage is None:
            self.test = Smiles2SmilesInPlaceDataset(
                self.src_test_path, self.tgt_test_path, self.max_size, self.tokenizer
            )

        if stage == "predict" or stage is None:
            self.prd = Smiles2SmilesInPlaceDataset(
                self.src_test_path, self.tgt_test_path, self.max_size, self.tokenizer
            )

    def collate_fn_default(
        self, batch: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        src_tokens, tgt_tokens = zip(*batch)
        src_tokens = torch.vstack(src_tokens)
        tgt_tokens = torch.vstack(tgt_tokens)
        return {"src_tokens": src_tokens, "tgt_tokens": tgt_tokens}

    def collate_fn_rotation(
        self, batch: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        src_tokens, tgt_tokens = zip(*batch)
        src_tokens = torch.vstack(src_tokens)
        tgt_tokens = torch.vstack(tgt_tokens)
        rotation_target = get_vocab_rotation_target(
            src_tokens, tgt_tokens, self.tokenizer.n_tokens
        )
        return {
            "src_tokens": src_tokens,
            "tgt_tokens": tgt_tokens,
            "rotation_target": rotation_target,
        }

    def train_dataloader(self):
        if self.rotation_target:
            collate_fn = self.collate_fn_rotation
            print("Using rotation target")
        else:
            collate_fn = self.collate_fn_default
            print("Using default target")

        if self.tokens_in_batch is None:
            return DataLoader(
                self.train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                persistent_workers=self.persistent_workers,
                pin_memory=self.pin_memory,
                shuffle=self.shuffle_train,
            )
        return DataLoader(
            self.train,
            batch_sampler=TokenSampler(
                self.train.target_lengths,
                self.tokens_in_batch,
                shuffle=self.shuffle_train,
            ),
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        if self.rotation_target:
            collate_fn = self.collate_fn_rotation
            print("Using rotation target")
        else:
            collate_fn = self.collate_fn_default
            print("Using default target")

        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        if self.rotation_target:
            collate_fn = self.collate_fn_rotation
            print("Using rotation target")
        else:
            collate_fn = self.collate_fn_default
            print("Using default target")

        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self):
        if self.rotation_target:
            collate_fn = self.collate_fn_rotation
            print("Using rotation target")
        else:
            collate_fn = self.collate_fn_default
            print("Using default target")

        return DataLoader(
            self.prd,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


class Smiles2CanonSmilesInPlaceDataset(Dataset):
    def __init__(
        self,
        smi_path: Path | str,
        tgt_path: Path | str,
        max_size: int,
        tokenizer: InplaceSMILESTokenizer,
    ):
        self.tokenizer = tokenizer
        self.max_size = max_size
        with open(src_path) as fs, open(tgt_path) as ft:
            self.source: list[str] = [s.strip() for s in fs.readlines()][:]
            self.target: list[str] = [s.strip() for s in ft.readlines()][:]
        assert len(self.source) == len(self.target), (
            f"The source and target data at {src_path} and {tgt_path} have different lenghts"
        )
        self.source_tokens = [self.tokenizer.encode(s) for s in self.source]
        self.target_tokens = [self.tokenizer.encode(s) for s in self.target]

        self.source_lengths = [len(i) for i in self.source_tokens]
        self.target_lengths = [len(i) for i in self.target_tokens]

    def __len__(self):
        return len(self.source_tokens)

    def __getitem__(self, item):
        src = torch.tensor(self.source_tokens[item]).long()
        tgt = torch.tensor(self.target_tokens[item]).long()

        src = pad(
            src,
            (0, self.max_size - src.size(-1)),
            "constant",
            self.tokenizer.pad_token_idx,
        )
        tgt = pad(
            tgt,
            (0, self.max_size - tgt.size(-1)),
            "constant",
            self.tokenizer.pad_token_idx,
        )

        return src, tgt


class Smiles2CanonSmilesInPlaceDM(LightningDataModule):
    """
    A Lightning Data Module specific to non-autoregressive SMILES-to-SMILES conversion.

    Args:
        :param data_dir: Path to the directory containing the data files in the format {src,tgt}-{train,val,test}.txt.
        If this format is not used, the paths to the source and target training, validation, and test data files can be provided explicitly.
        :param src_train_path: Optional path to the source training data file.
        :param tgt_train_path: Optional path to the target training data file.
        :param src_val_path: Optional path to the source validation data file.
        :param tgt_val_path: Optional path to the target validation data file.
        :param src_test_path: Optional path to the source test data file.
        :param tgt_test_path: Optional path to the target test data file.
        :param vocab_path: Path to the vocabulary file.
        :param max_size: Maximum size of the input/output sequences.
        The sequences in batches will are padded to have this length.
        :param rotation_target: If true, the model will predict shifts in tokens between source and target
        instead of predicting the target tokens directly.
        :param batch_size: Number of samples per batch.
        :param tokens_in_batch: Maximum number of tokens in a batch. If provided, batching will try to minimize
        the proportion of pad token in batches by having sequences of similar length in evety batch.
        :param num_workers: Number of subprocesses to use for data loading.
        :param persistent_workers: Whether to maintain the workers between iterations.
        :param pin_memory: Whether to pin memory in data loader.
        :param shuffle_train: Whether to shuffle the training data.
    """

    def __init__(
        self,
        data_dir: str | None = None,  # Data location arguments
        src_train_path: str | None = None,
        tgt_train_path: str | None = None,
        src_val_path: str | None = None,
        tgt_val_path: str | None = None,
        src_test_path: str | None = None,
        tgt_test_path: str | None = None,
        vocab_path: str | None = None,
        max_size: int = 200,
        rotation_target: bool = False,
        batch_size: int = 1,  # Batching arguments
        tokens_in_batch: int | None = None,
        num_workers: int = 0,
        persistent_workers=False,
        pin_memory: bool = False,
        shuffle_train: bool = False,
    ):
        super().__init__()

        self.data_dir = Path(data_dir).resolve()
        self.vocab_path = vocab_path
        self.max_size = max_size
        self.rotation_target = rotation_target
        self.src_train_path = src_train_path or self.data_dir / "src-train.txt"
        self.tgt_train_path = tgt_train_path or self.data_dir / "tgt-train.txt"
        self.src_val_path = src_val_path or self.data_dir / "src-val.txt"
        self.tgt_val_path = tgt_val_path or self.data_dir / "tgt-val.txt"
        self.src_test_path = src_test_path or self.data_dir / "src-test.txt"
        self.tgt_test_path = tgt_test_path or self.data_dir / "tgt-test.txt"

        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.tokens_in_batch = tokens_in_batch

        self.tokenizer = self.create_tokenizer()

    def create_tokenizer(self):
        if self.vocab_path is None:
            self.vocab_path = self.data_dir / "vocabs" / "vocab.json"
        else:
            self.vocab_path = Path(self.vocab_path).resolve()

        tokenizer = InplaceSMILESTokenizer()
        try:
            tokenizer.load_vocab(self.vocab_path)
            print(f"Loaded tokenizer vocabulary from {self.vocab_path}")
        except FileNotFoundError:
            print("Training tokenizer...")
            with open(self.src_train_path) as f, open(self.tgt_train_path) as g:
                tokenizer.train_tokenizer(chain(f, g))
            tokenizer.save_vocab(self.vocab_path)
            print(f"Saved tokenizer vocab to: {self.vocab_path}")
        finally:
            return tokenizer

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train = Smiles2SmilesInPlaceDataset(
                self.src_train_path, self.tgt_train_path, self.max_size, self.tokenizer
            )
            self.val = Smiles2SmilesInPlaceDataset(
                self.src_val_path, self.tgt_val_path, self.max_size, self.tokenizer
            )

        if stage == "validate":
            self.val = Smiles2SmilesInPlaceDataset(
                self.src_val_path, self.tgt_val_path, self.max_size, self.tokenizer
            )

        if stage == "test" or stage is None:
            self.test = Smiles2SmilesInPlaceDataset(
                self.src_test_path, self.tgt_test_path, self.max_size, self.tokenizer
            )

        if stage == "predict" or stage is None:
            self.prd = Smiles2SmilesInPlaceDataset(
                self.src_test_path, self.tgt_test_path, self.max_size, self.tokenizer
            )

    def collate_fn_default(
        self, batch: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        src_tokens, tgt_tokens = zip(*batch)
        src_tokens = torch.vstack(src_tokens)
        tgt_tokens = torch.vstack(tgt_tokens)
        return {"src_tokens": src_tokens, "tgt_tokens": tgt_tokens}

    def collate_fn_rotation(
        self, batch: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        src_tokens, tgt_tokens = zip(*batch)
        src_tokens = torch.vstack(src_tokens)
        tgt_tokens = torch.vstack(tgt_tokens)
        rotation_target = get_vocab_rotation_target(
            src_tokens, tgt_tokens, self.tokenizer.n_tokens
        )
        return {
            "src_tokens": src_tokens,
            "tgt_tokens": tgt_tokens,
            "rotation_target": rotation_target,
        }

    def train_dataloader(self):
        if self.rotation_target:
            collate_fn = self.collate_fn_rotation
            print("Using rotation target")
        else:
            collate_fn = self.collate_fn_default
            print("Using default target")

        if self.tokens_in_batch is None:
            return DataLoader(
                self.train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                persistent_workers=self.persistent_workers,
                pin_memory=self.pin_memory,
                shuffle=self.shuffle_train,
            )
        return DataLoader(
            self.train,
            batch_sampler=TokenSampler(
                self.train.target_lengths,
                self.tokens_in_batch,
                shuffle=self.shuffle_train,
            ),
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        if self.rotation_target:
            collate_fn = self.collate_fn_rotation
            print("Using rotation target")
        else:
            collate_fn = self.collate_fn_default
            print("Using default target")

        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        if self.rotation_target:
            collate_fn = self.collate_fn_rotation
            print("Using rotation target")
        else:
            collate_fn = self.collate_fn_default
            print("Using default target")

        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self):
        if self.rotation_target:
            collate_fn = self.collate_fn_rotation
            print("Using rotation target")
        else:
            collate_fn = self.collate_fn_default
            print("Using default target")

        return DataLoader(
            self.prd,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
