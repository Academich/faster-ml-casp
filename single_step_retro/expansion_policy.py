import torch

from single_step_retro.src.model.medusa import SmilesToSmilesAutoregressiveMedusaModel
from single_step_retro.src.data_handling.tokenization import InplaceSMILESTokenizer

CKPT = "/home/mikhail/work/SmilesToSmilesNA/checkpoints/medusa/Retro/new_metrics_3e_4_from_scratch/cntnet_lr_00003/step=7827-val_l=0.249-val/acc_sequence_head_0=0.142-val/sum_beam_srch_9_acc=0.892.ckpt"
VOCAB_PATH="/home/mikhail/work/SmilesToSmilesNA/data/USPTO_50K_PtoR_aug1/test/vocabs/vocab.json"

tokenizer = InplaceSMILESTokenizer()
tokenizer.load_vocab("smiles_medusa_vocab.json")
model = SmilesToSmilesAutoregressiveMedusaModel(
    embedding_dim=256,
    feedforward_dim=2048,
    num_encoder_layers=6,
    num_decoder_layers=6,
    num_heads=8,
    activation="gelu",
    tokenizer=tokenizer,
    max_size=200,
    beam_size=5
)

checkpoint = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["state_dict"], strict=True)
model.eval()
smiles = "c1ccccc1CCc1ccccc1C"
tgt_smi="CC(C)(C)OC(=O)OC(=O)OC(C)(C)C.Cc1ccc(S(=O)(=O)O[C@@H]2CN[C@H]3[C@@H]2OC[C@@H]3O)cc1"

tokens = torch.tensor(tokenizer.encode(smiles)).long().unsqueeze(0)
print(tokens)
generated = model.generate(tokens)
print(generated)
gen_smiles = tokenizer.decode_batch(generated[0].cpu().numpy())

print(gen_smiles)


