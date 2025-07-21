import argparse
import os
import sys
import uuid

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from external_modelzoo.ssbenchmark.ssmodels.base_model import Registries, SSMethod
from external_modelzoo.ssbenchmark.utils import canonicalize_smiles, split_reactions


@Registries.ModelChoice.register(key="transformer")
class model_transformer(SSMethod):
    def __init__(self, module_path=None):
        self.model_name = "Transformer"
        if module_path is not None:
            sys.path.insert(len(sys.path), os.path.abspath(module_path))

    def preprocess(self, data, reaction_col):
        data["reactants"], data["spectators"], data["products"] = split_reactions(
            data[reaction_col].tolist()
        )
        data["products"] = data.products.apply(canonicalize_smiles)
        data["reactants"] = data.reactants.apply(canonicalize_smiles)
        data["products_mol"] = [
            Chem.MolFromSmiles(m) if m is not None else None for m in data["products"]
        ]
        data["reactants_mol"] = [
            Chem.MolFromSmiles(m) if m is not None else None for m in data["reactants"]
        ]
        data["products_mol"] = [
            Chem.MolToSmiles(m) if m is not None else None for m in data["products_mol"]
        ]
        data["reactants_mol"] = [
            Chem.MolToSmiles(m) if m is not None else None
            for m in data["reactants_mol"]
        ]
        data = data.dropna()
        data = data.rename(columns={"class": "reaction_type", "split": "set"})
        if "reaction_type" not in data.columns:
            data["reaction_type"] = np.nan
        data = data[["reactants_mol", "products_mol", "reaction_type", "set"]]
        return data

    def process_input(self, data, reaction_col):
        print("preprocess input")
        data = self.preprocess(data, reaction_col)
        return data
        data = data.rename(columns={"class": "reaction_type", "split": "set"})
        return data[["reactants_mol", "products_mol", "reaction_type", "set"]]

    def preprocess_store(self, data, preprocess_root, instance_name):
        print("preprocess store")
        oroot = os.path.os.path.abspath(preprocess_root)
        if not self.check_path(oroot):
            self.make_path(oroot)
        data = data.reset_index(drop=True)
        data.to_pickle(os.path.join(oroot, f"{instance_name}.pickle"))
        data = data[data["set"] == "test"]
        data.to_csv(
            os.path.join(oroot, f"test_smiles_{instance_name}.txt"),
            index=False,
            header=False,
        )

    def process_output(self, data_root, instance_name, k):
        opath = data_root
        # opath = os.path.join(data_root, self.model_name.lower(), f"{instance_name}.pickle")
        if not self.check_path(opath):
            raise FileNotFoundError(f"File not found at {opath}")
        data = pd.read_pickle(opath)
        columns = [f"prediction_{i}" for i in range(k)]
        data = data[columns]
        data = data.values.tolist()

        return [[canonicalize_smiles(p) for p in pred] for pred in data]

    def build_dataset(args):
        text = Path(args.reactants_path).read_text()
        smiles = text.split("\n")
        smiles = [smi for smi in smiles if smi != "" and smi is not None]
        dataset = ReactionDataset(smiles, smiles)
        return dataset

    def model_setup(self, use_gpu=False, **kwargs):
        from single_step_retro.src.model.medusa import SmilesToSmilesAutoregressiveMedusaModel
        from single_step_retro.src.data_handling.tokenization import InplaceSMILESTokenizer
        
        self.device = "cuda:0" if use_gpu else "cpu"

        vocab_path = kwargs.get("vocab_path", None)
        checkpoint_path = kwargs.get("checkpoint_path", None)
        embedding_dim = kwargs.get("embedding_dim", 256)
        feedforward_dim = kwargs.get("feedforward_dim", 2048)
        num_encoder_layers = kwargs.get("num_encoder_layers", 6)
        num_decoder_layers = kwargs.get("num_decoder_layers", 6)
        num_heads = kwargs.get("num_heads", 8)
        activation = kwargs.get("activation", "relu")
        max_size = kwargs.get("max_size", 200)
        beam_size = kwargs.get("beam_size", 5)

        tokenizer = InplaceSMILESTokenizer()
        tokenizer.load_vocab(vocab_path)
        model = SmilesToSmilesAutoregressiveMedusaModel(
            embedding_dim=embedding_dim,
            feedforward_dim=feedforward_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            activation=activation,
            tokenizer=tokenizer,
            max_size=max_size,
            beam_size=beam_size
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        print("NATASHA")
        model.eval()
        self.model = model


    def build_datamodule(self, args, dataset, tokeniser, max_seq_len):
        from molbart.data.datamodules import FineTuneReactionDataModule

        test_idxs = range(len(dataset))
        dm = FineTuneReactionDataModule(
            dataset,
            tokeniser,
            args.batch_size,
            max_seq_len,
            val_idxs=[],
            test_idxs=test_idxs,
        )
        return dm

    def predict(self, model, test_loader):
        model = model.to(self.device)
        model.eval()

        smiles = []
        log_lhs = []
        original_smiles = []

        for b_idx, batch in enumerate(test_loader):
            device_batch = {
                key: val.to(self.device) if type(val) == torch.Tensor else val
                for key, val in batch.items()
            }
            with torch.no_grad():
                smiles_batch, log_lhs_batch = model.sample_molecules(
                    device_batch, sampling_alg="beam"
                )

            smiles.extend(smiles_batch)
            log_lhs.extend(log_lhs_batch)
            original_smiles.extend(batch["target_smiles"])

        return smiles, log_lhs, original_smiles

    def _model_call(self, X):
        from molbart.data.datasets import ReactionDataset
        from molbart.predict import predict

        print("Reading dataset...")
        dataset = ReactionDataset(X, X)
        print("Finished dataset.")

        dm = self.build_datamodule(
            self.model_args, dataset, self.tokeniser, self.model.max_seq_len
        )
        dm.setup()
        test_loader = dm.test_dataloader()
        print("Finished loader.")

        print("Evaluating model...")
        smiles, log_lhs, original_smiles = self.predict(self.model, test_loader)
        output = F.softmax(
            torch.Tensor(log_lhs), dim=1
        )  # Added by Paula: AZF requires probabilities not log_lhs

        return smiles, output.tolist()

    def model_call(self, X):
        return self._model_call(X)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module_path", type=str, default=None)
    parser.add_argument("--preprocess_batch", type=str, default=None)
    parser.add_argument("--preprocess_join", type=str, default=None)
    parser.add_argument("--reaction_col", type=str, default="reaction_smiles")
    parser.add_argument("--preprocess_output", type=str, default=None)
    parser.add_argument("--instance_name", type=str, default=None)

    args = parser.parse_args()

    model_instance = model_transformer(module_path=args.module_path)
    if args.preprocess_batch is not None:
        df = model_instance.read_csv(args.preprocess_batch)
        df = model_instance.preprocess(df, args.reaction_col)
        model_instance.write_pickle(df, args.preprocess_output, str(uuid.uuid4()))

    if args.preprocess_join is not None:
        df = model_instance.gather_batches(args.preprocess_join, "pickle")
        model_instance.preprocess_store(df, args.preprocess_output, args.instance_name)
