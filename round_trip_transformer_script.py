"""
This script evaluates round-trip accuracy and other metrics of single-step retrosynthesis models on a test dataset (e.g., USPTO 50k)
"""
import yaml
import argparse

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from external_modelzoo.ssbenchmark.utils import canonicalize_smiles


def model_setup(use_gpu, **kwargs):
    sampling = kwargs.get("sampling")
    if sampling is None:
        raise ValueError("'sampling' is required. Choose from 'beam_search', 'medusa'")
    if sampling == "medusa":
        from single_step_retro.src.model.medusa import SmilesToSmilesAutoregressiveMedusaModel
        from single_step_retro.src.data_handling.tokenization import InplaceSMILESTokenizer
        model_class = SmilesToSmilesAutoregressiveMedusaModel
    elif sampling == "beam_search_speculative":
        from single_step_retro.src.model.autoregressive import VanillaEncoderDecoderTransformerLightning
        from single_step_retro.src.data_handling.tokenization import InplaceSMILESTokenizer
        model_class = VanillaEncoderDecoderTransformerLightning
    elif sampling == "beam_search":
        from single_step_retro.src.model.autoregressive import VanillaEncoderDecoderTransformerLightning
        from single_step_retro.src.data_handling.tokenization import InplaceSMILESTokenizer
        model_class = VanillaEncoderDecoderTransformerLightning
    else:
        raise ValueError(f"Invalid sampling method: {sampling}")

    device = "cuda:0" if use_gpu else "cpu"

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
    model = model_class(
        embedding_dim=embedding_dim,
        feedforward_dim=feedforward_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        activation=activation,
        tokenizer=tokenizer,
        sampling=sampling,
        max_len=max_size,
        beam_size=beam_size
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    return model


def unicalize(curr_rxn_smiles_l, curr_probs_l):
    uniq_smis = set()
    unic_smiles_list = []
    probs_list = []

    for can_smi, prob in zip(curr_rxn_smiles_l, curr_probs_l):
        if can_smi is not None and can_smi not in uniq_smis:
            uniq_smis.add(can_smi)
            unic_smiles_list.append(can_smi)
            probs_list.append(prob)

    return unic_smiles_list, probs_list


def filter(curr_rxn_smiles, curr_probs):
    canonicalize_smiles_list = [canonicalize_smiles(smi, sort=True) for smi in curr_rxn_smiles]

    filtered_smiles_list, probs_list = unicalize(canonicalize_smiles_list, curr_probs)
    return filtered_smiles_list, probs_list


def round_trip_transformer_or_med(retro_model, product_model, test_path,
                                  retro_model_batch_size):  # X is a list of B smiles
    with open(test_path) as fs:
        desired_mols_list = [s.strip() for s in fs.readlines()]

    # RB is short for retro_model_batch_size
    batches_of_tgt_smis = [desired_mols_list[i:i + retro_model_batch_size] for i in
                           range(0, len(desired_mols_list), retro_model_batch_size)]

    eff_N = []
    round_trip_percent = []
    round_trip_pcs = []
    prod_in_reactnts_pcs = []
    small_r_problems = []

    for smi_batch in tqdm(batches_of_tgt_smis):
        tgt_canon_smi_list = [canonicalize_smiles(psmi, sort=True) for psmi in smi_batch]
        tgt_canon_smi_list = [psmi for psmi in tgt_canon_smi_list if psmi is not None]

        if len(tgt_canon_smi_list) > 0:
            list_of_tensors_for_retro_model = [
                torch.tensor(retro_model.tokenizer.encode(smi), device=retro_model.device).long() for smi in
                tgt_canon_smi_list]
            input_tensor_for_retro_model = pad_sequence(list_of_tensors_for_retro_model,
                                                        padding_value=retro_model.tokenizer.pad_token_idx,
                                                        batch_first=True)  # -> (RB,L)

            with torch.inference_mode():
                rxn_generated, rxn_probs = retro_model.generate(
                    input_tensor_for_retro_model)  # -> (RB, RK, L), (RB, RK)

        for i, tgt_smi in enumerate(smi_batch):
            rxns_for_tgt_smi = retro_model.tokenizer.decode_batch(rxn_generated[i].cpu().numpy())  # list of RK smiles
            probs_for_tgt_smi = rxn_probs[i].cpu().numpy()  # -> (RK)

            unic_canon_rxns_for_tgt_smi, unic_canon_probs_for_tgt_smi = filter(rxns_for_tgt_smi, probs_for_tgt_smi)
            eff_N.append(len(unic_canon_rxns_for_tgt_smi))  # canon smis without repetitions; eff_n <= RK

            if len(unic_canon_rxns_for_tgt_smi) == 0:
                continue

            list_of_tokenized_rxns_for_prod_model = [
                torch.tensor(product_model.tokenizer.encode(rxn), device=product_model.device).long() for rxn in
                unic_canon_rxns_for_tgt_smi]
            input_tensor_for_prod_model = pad_sequence(list_of_tokenized_rxns_for_prod_model,
                                                       padding_value=product_model.tokenizer.pad_token_idx,
                                                       batch_first=True)  # -> (eff_n, L)
            with torch.inference_mode():
                prod_model_generated, prod_model_probs = product_model.generate(
                    input_tensor_for_prod_model)  ## -> (eff_n, PK, L), (eff_n, PK)

            num_of_rxns_for_tgt_smi = prod_model_generated.shape[0]  # (eff_n)
            round_trip = []

            for j in range(num_of_rxns_for_tgt_smi):
                product_smis_for_rxnj = product_model.tokenizer.decode_batch(
                    prod_model_generated[j].cpu().numpy())  # list of PK prods

                canon_product_smis_for_rxnj = [canonicalize_smiles(psmi, sort=True) for psmi in product_smis_for_rxnj]
                canon_product_smis_for_rxnj = [psmi for psmi in canon_product_smis_for_rxnj if psmi is not None]

                round_trip.append(tgt_smi in canon_product_smis_for_rxnj)

                rxnj_smi = unic_canon_rxns_for_tgt_smi[j]
                reactants_of_rxnj = rxnj_smi.split(".")
                prod_in_reactnts_pcs.append(tgt_smi in reactants_of_rxnj)

                if list_of_tokenized_rxns_for_prod_model[j].shape[-1] < 8:
                    print("WARNING!small reactants", rxnj_smi + ">>" + tgt_smi)
                    small_r_problems.append(rxnj_smi + ">>" + tgt_smi)

            round_trip_percent.append(sum(round_trip) / len(round_trip))
            round_trip_pcs.append(sum(round_trip))

    print("Av. round-trip, % ", 100 * sum(round_trip_percent) / len(round_trip_percent))
    print("Av. round-trip, reactions ", sum(round_trip_pcs) / len(round_trip_pcs))
    print("Av. # valid unique reactions, reactions ", sum(eff_N) / len(eff_N))

    print("Reactions with product in reactants, reactions", sum(prod_in_reactnts_pcs))
    print("Less than 8 tokens in reactants problem, reactions", len(set(small_r_problems)))


def main(retro_model_config, product_model_config, test_path, use_gpu, retro_model_batch_size):
    with open(retro_model_config) as stream:
        try:
            retro_model_kwargs = yaml.safe_load(stream)['expansion']['transformer']['data']['kwargs']
            print(retro_model_kwargs)
            retro_model = model_setup(use_gpu, **retro_model_kwargs)
        except yaml.YAMLError as exc:
            print(exc)
    with open(product_model_config) as stream:
        try:
            prod_model_kwargs = yaml.safe_load(stream)['expansion']['transformer']['data']['kwargs']
            print(prod_model_kwargs)
            product_model = model_setup(use_gpu, **prod_model_kwargs)
        except yaml.YAMLError as exc:
            print(exc)

    round_trip_transformer_or_med(retro_model, product_model, test_path, retro_model_batch_size)


if __name__ == "__main__":
    from rdkit import RDLogger

    RDLogger.DisableLog('rdApp.*')

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", "-t", type=str, default="data/uspto50k_test/products.txt",
                        help="path to USPTO 50k test")
    parser.add_argument("--product_model_config", "-pyml", type=str,
                        default="configs/transformer_product_model_config.yml")

    parser.add_argument("--retro_model_config", "-ryml", type=str,
                        default="configs/medusa_default_config.yml",
                        help="path to config of evaluated single-step retrosynthesys model (Medusa or Transformer)")
    parser.add_argument("--use_gpu", action="store_true",
                        help="whether to use GPU (cuda:0). Run with CUDA_VISIBLE_DEVICES to select GPUs other than 0 by default")
    parser.add_argument("--retro_model_batch_size", "-bsz", type=int, default=32,
                        help="batch size of the single-step retro model")
    args = parser.parse_args()

    main(args.retro_model_config, args.product_model_config, args.test_path, args.use_gpu, args.retro_model_batch_size)
