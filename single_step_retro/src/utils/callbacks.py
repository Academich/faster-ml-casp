"""
Custom callbacks for pytorch lightning.
Used for decoding the predicted token indices into strings.
"""

from pathlib import Path
from typing import Any

from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT

from wandb import Table as wb_Table


class DecodingCallback(Callback):
    """
    A callback that decodes the predicted token indices
    into strings on validation and logs the results to the logger.
    """

    def __init__(self):
        super().__init__()
        self.validation_step_outputs = []

    def on_validation_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.validation_step_outputs.append(outputs)

    def on_validation_epoch_end(
        self, trainer: "Trainer", pl_module: "LightningModule"
    ) -> None:
        tkz = pl_module.tokenizer
        total_correct, total = 0, 0
        n_samples_logged, n_samples_to_log = 0, 5
        decoded_samples = []
        for o in self.validation_step_outputs:
            pred_tokens = o["pred_tokens"].cpu().numpy()
            target_ahead = o["target_ahead"].cpu().numpy()
            b_size = pred_tokens.shape[0]
            for i in range(b_size):
                target_str = tkz.decode(target_ahead[i])
                predicted_str = tkz.decode(pred_tokens[i])
                total_correct += int(predicted_str == target_str)
                total += 1
                if n_samples_logged < n_samples_to_log:
                    decoded_samples.append((target_str, predicted_str))
                    n_samples_logged += 1

        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.log_metrics(
                    {"val/whole_seq_exact_match_acc_total": total_correct / total}
                )
            elif isinstance(logger, WandbLogger):
                logger.experiment.log(
                    {"val/whole_seq_exact_match_acc_total": total_correct / total}
                )
                logger.experiment.log(
                    {
                        "val/decoded_samples": wb_Table(
                            columns=["target", "prediction"], data=decoded_samples
                        )
                    }
                )

        self.validation_step_outputs.clear()


class PredictionWriter(BasePredictionWriter):
    """
    A callback that writes the predicted token indices into strings
    and saves them to a file in the prediction (inference) phase.
    """

    def __init__(self, output_file_path, write_interval="batch"):
        super().__init__(write_interval)
        self.output_path = Path(output_file_path).resolve()
        self.output_path.unlink(missing_ok=True)
        self.output_path.parent.mkdir(exist_ok=True)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        tkz = pl_module.tokenizer
        prediction_np = prediction.cpu().numpy()

        _, n_predictions, _ = prediction_np.shape
        with open(self.output_path, "a") as f:
            if f.tell() == 0:
                print(
                    ",".join(
                        ["source", "target"]
                        + [f"prediction_{i}" for i in range(1, n_predictions + 1)]
                    ),
                    file=f,
                )
            src = batch["src_tokens"].cpu().numpy()
            tgt = batch["tgt_tokens"].cpu().numpy()
            for i, (s, t) in enumerate(zip(src, tgt)):
                s_string = tkz.decode(s)
                t_string = tkz.decode(t)
                p_options = tkz.decode_batch(prediction_np[i])
                print(",".join([s_string, t_string] + p_options), file=f)
