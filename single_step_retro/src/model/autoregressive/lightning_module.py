"""
PyTorch Lightning modules with training, validation,
testing and prediction loops and optimizers.
"""

import json
import datetime
from timeit import default_timer as timer
from typing import Any
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, optim

from single_step_retro.src.data_handling.tokenization import InplaceSMILESTokenizer
from single_step_retro.src.model.autoregressive.modules import VanillaTransformer
from single_step_retro.src.decoding.standard_decoding import TranslationInferenceBeamSearch, TranslationInferenceGreedy
from single_step_retro.src.decoding.speculative_decoding import TranslationInferenceBeamSearchSpeculative, TranslationInferenceGreedySpeculative

from single_step_retro.src.utils.metrics import calc_sequence_acc, calc_token_acc
from single_step_retro.src.utils.lr_schedule import NoamLRSchedule, ConstantLRSchedule


class VanillaEncoderDecoderTransformerLightning(pl.LightningModule):
    def __init__(
        self,
        embedding_dim: int = 128,  # Model arguments
        feedforward_dim: int = 256,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        num_heads: int = 4,
        dropout_rate: float = 0.0,
        activation: str = "relu",

        learning_rate: float = 3e-4,  # Optimization arguments
        weight_decay: float = 0.0,
        scheduler: str = "const",
        warmup_steps: int = 0,

        tokenizer: InplaceSMILESTokenizer | None = None,
        sampling: str = "beam_search",  # Prediction generation arguments
        beam_size: int = 0,
        max_len: int = 0,
        n_drafts: int = 0,
        draft_len: int = 0,
        smart_drafts_mode: bool = True,
        report_prediction_time: bool = True,
        report_prediction_file: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["src_tokenizer", "tgt_tokenizer"])

        assert tokenizer is not None, "source tokenizer not provided"
        assert tokenizer is not None, "target tokenizer not provided"
        self.src_tokenizer = tokenizer
        self.tgt_tokenizer = tokenizer
        self.tokenizer = tokenizer
        self.src_vocab_size: int = len(tokenizer)
        self.tgt_vocab_size: int = len(tokenizer)
        self.src_pad_token_i: int = tokenizer.pad_token_idx
        self.src_bos_token_i: int = tokenizer.bos_token_idx
        self.src_eos_token_i: int = tokenizer.eos_token_idx
        self.tgt_pad_token_i: int = tokenizer.pad_token_idx
        self.tgt_bos_token_i: int = tokenizer.bos_token_idx
        self.tgt_eos_token_i: int = tokenizer.eos_token_idx

        self.criterion = nn.CrossEntropyLoss(reduction="mean")

        self.model = self._create_model()

        self.generator = self._create_generator()
        print(self.generator)

        self.report_prediction_time = report_prediction_time
        self.prediction_start_time = None

    def _create_model(self) -> torch.nn.Module:
        model = VanillaTransformer(
            self.src_vocab_size,
            self.tgt_vocab_size,
            self.hparams.num_encoder_layers,
            self.hparams.num_decoder_layers,
            self.hparams.embedding_dim,
            self.hparams.num_heads,
            self.hparams.feedforward_dim,
            self.hparams.dropout_rate,
            self.hparams.activation,
            share_embeddings=False,
            src_pad_token_idx=self.src_pad_token_i,
            tgt_pad_token_idx=self.tgt_pad_token_i,
        )
        return model

    def _create_generator(self):
        if self.hparams.sampling == "greedy":
            return TranslationInferenceGreedy(
                self.model,
                max_len=self.hparams.max_len,
                pad_token=self.tgt_pad_token_i,
                bos_token=self.tgt_bos_token_i,
                eos_token=self.tgt_eos_token_i,
            )
        elif self.hparams.sampling == "beam_search":
            return TranslationInferenceBeamSearch(
                self.model,
                beam_size=self.hparams.beam_size,
                max_len=self.hparams.max_len,
                pad_token=self.tgt_pad_token_i,
                bos_token=self.tgt_bos_token_i,
                eos_token=self.tgt_eos_token_i,
            )

        elif self.hparams.sampling == "greedy_speculative":
            assert self.hparams.draft_len > 0, (
                "Number of speculative tokens must be a positive integer."
            )
            return TranslationInferenceGreedySpeculative(
                self.model,
                max_len=self.hparams.max_len,
                draft_len=self.hparams.draft_len,
                n_drafts=self.hparams.n_drafts,
                pad_token=self.tgt_pad_token_i,
                bos_token=self.tgt_bos_token_i,
                eos_token=self.tgt_eos_token_i,
                replace_token=self.tgt_tokenizer.encoder_dict["c"],
            )
        elif self.hparams.sampling == "beam_search_speculative":
            return TranslationInferenceBeamSearchSpeculative(
                self.model,
                vocab_size=self.tgt_vocab_size,
                max_len=self.hparams.max_len,
                n_best=self.hparams.beam_size,
                draft_len=self.hparams.draft_len,
                n_drafts=self.hparams.n_drafts,
                pad_token=self.tgt_pad_token_i,
                bos_token=self.tgt_bos_token_i,
                eos_token=self.tgt_eos_token_i,
                C_token=self.tgt_tokenizer.encoder_dict["c"],
                smart_drafts_mode=self.hparams.smart_drafts_mode,
            )

        else:
            options = ", ".join(
                [
                    "beam_search",
                    "greedy",
                    "greedy_speculative",
                    "beam_search_speculative",
                ]
            )
            raise ValueError(
                f"Unknown sampling option {self.hparams.sampling}. Options are {options}."
            )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        The model receives source token indices and target token indices,
        and possibly some masks or any other necessary information.
        The output is the predicted next token probability distribution,
        a tensor of shape BATCH_SIZE x SEQUENCE_LENGTH x TARGET_VOCABULARY_SIZE
        """
        return self.model(batch["src_tokens"], batch["tgt_tokens"][:, :-1])

    def _calc_loss(self, logits, tgt_ids):
        _, _, tgt_vocab_size = logits.size()
        return self.criterion(logits.reshape(-1, tgt_vocab_size), tgt_ids.reshape(-1))

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        pred_logits = self.__call__(batch)

        # We predict the next token given the previous ones
        target_future = batch["tgt_tokens"][:, 1:]

        loss = self._calc_loss(pred_logits, target_future)

        pred_tokens = torch.argmax(pred_logits, dim=2)
        token_acc = calc_token_acc(pred_tokens, target_future)
        sequence_acc = calc_sequence_acc(
            pred_tokens, target_future, self.tgt_eos_token_i
        )
        mean_pad_tokens_in_target = (
            (target_future == self.tgt_pad_token_i).float().mean()
        )

        self.log(f"train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            f"train/acc_single_tok",
            token_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"train/acc_sequence",
            sequence_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"train/pads_in_batch_tgt",
            mean_pad_tokens_in_target,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        pred_logits = self.__call__(batch)

        # We predict the next token given the previous ones
        target_future = batch["tgt_tokens"][:, 1:]

        loss = self._calc_loss(pred_logits, target_future)

        pred_tokens = torch.argmax(pred_logits, dim=2)
        token_acc = calc_token_acc(pred_tokens, target_future)
        sequence_acc = calc_sequence_acc(
            pred_tokens, target_future, self.tgt_eos_token_i
        )

        self.log(f"val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            f"val/acc_single_tok",
            token_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"val/acc_sequence",
            sequence_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return {"pred_tokens": pred_tokens, "target_ahead": target_future}

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        pred_logits = self.__call__(batch)

        source = batch["src_tokens"]
        target = batch["tgt_tokens"]
        target_future = batch["tgt_tokens"][:, 1:]

        loss = self._calc_loss(pred_logits, target_future)
        pred_tokens = torch.argmax(pred_logits, dim=2)
        token_acc = calc_token_acc(pred_tokens, target_future)
        sequence_acc = calc_sequence_acc(
            pred_tokens, target_future, self.tgt_eos_token_i
        )

        self.log(f"test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            f"test/acc_single_tok",
            token_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"test/acc_sequence",
            sequence_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        return {
            "source_token_ids": source,
            "pred_logits": pred_logits,
            "target_token_ids": target,
        }
    
    def generate(self, src_token_ids: torch.LongTensor) -> list[torch.LongTensor]:
        return self.generator.generate(src_token_ids)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        source = batch["src_tokens"]
        generated = self.generate(source)
        return generated

    def on_predict_start(self) -> None:
        if self.report_prediction_time:
            self.prediction_start_time = timer()

    def on_predict_end(self) -> None:
        if self.report_prediction_time:
            elapsed = datetime.timedelta(seconds=timer() - self.prediction_start_time)
            report = {
                "algorithm": self.hparams.sampling,
                "batch_size": self.trainer.datamodule.batch_size,
                "tgt_test_path": str(self.trainer.datamodule.tgt_test_path),
                "max_len": self.hparams.max_len,
                "total_seconds": round(elapsed.total_seconds(), 4),
                "model_calls": self.generator.model_calls_num,
                "seconds_per_model_call": round(
                    elapsed.total_seconds() / self.generator.model_calls_num, 4
                ),
            }
            if self.hparams.sampling in (
                "greedy_speculative",
                "beam_search_speculative",
            ):
                report["n_drafts"] = self.hparams.n_drafts
                report["draft_len"] = self.hparams.draft_len
                if self.hparams.sampling == "beam_search_speculative":
                    report["accepted_tokens"] = self.generator.accepted_tokens_num
                    report["acceptance_rate"] = round(
                        self.generator.accepted_tokens_num
                        / self.generator.produced_non_pad_tokens,
                        4,
                    )
            report = json.dumps(report)

            print(report)
            if self.hparams.report_prediction_file is not None:
                report_prediction_dir = Path(self.hparams.report_prediction_file).parent
                report_prediction_dir.mkdir(exist_ok=True)
                with open(self.hparams.report_prediction_file, "a") as f:
                    print(report, file=f)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
        )

        sched_name = self.hparams.scheduler
        ws = self.hparams.warmup_steps
        if sched_name == "const":
            scheduler = {
                "scheduler": optim.lr_scheduler.LambdaLR(
                    optimizer, ConstantLRSchedule(ws)
                ),
                "name": "Constant LR scheduler",
                "interval": "step",
                "frequency": 1,
            }
        elif sched_name == "noam":
            d = (
                self.model.emb_dim
            )  # May fail if the model does not have an 'emb_dim' attribute
            scheduler = {
                "scheduler": optim.lr_scheduler.LambdaLR(
                    optimizer, NoamLRSchedule(d, ws)
                ),
                "name": "Noam scheduler",
                "interval": "step",
                "frequency": 1,
            }
        else:
            raise ValueError(
                f'Unknown scheduler name {self.hparams.scheduler}. Options are "const", "noam".'
            )

        return [optimizer], [scheduler]