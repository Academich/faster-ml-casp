"""
This is the runnable entry point for training and inference of pytorch lightning models.
Most conviniently configured using a config (YAML) file.
This file is not supposed to be changed.
"""

from pytorch_lightning.cli import LightningCLI


class TranslationCLI(LightningCLI):
    """
    A CLI that allows using subcommands together with run=False.
    """

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.tokenizer", "model.init_args.tokenizer", apply_on="instantiate")


if __name__ == "__main__":
    cli = TranslationCLI(
        model_class=None,  # the model class should be provided in the CLI
        datamodule_class=None,  # the data module class should be provided in the CLI
        save_config_kwargs={"overwrite": True},
        run=True,
    )
