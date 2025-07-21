from torch import LongTensor, Tensor


def calc_token_acc(pred_ids: LongTensor, tgt_ids: LongTensor, ignored_tgt_index=None) -> Tensor:
    """
    Calculates the prediction accuracy for single tokens in a batch.
    The ignored_tgt_index (if it is not None) doesn't take part in calculating
    of the token accuracy
    B = batch size, L = target sequence length
    Args:
        ignored_tgt_index: int
        pred_ids: (B, L)
        tgt_ids: (B, L)
    Returns:
        Tensor of one element
    """
    single_tokens_predicted_right = (pred_ids == tgt_ids).float()
    if ignored_tgt_index is not None:
        bool_mask_t = tgt_ids == ignored_tgt_index
        single_tokens_predicted_right.masked_fill_(bool_mask_t, 0.)
        return single_tokens_predicted_right.sum() / (~bool_mask_t).sum()
    return single_tokens_predicted_right.mean()


def calc_sequence_acc(
    pred_ids: LongTensor, tgt_ids: LongTensor, eos_token_idx: int
) -> Tensor:
    """
    Checks how many sequences in a batch are predicted perfectly.
    Considers only the tokens before the first end-of-sequence token.
    Intended for use with autoregressive predictions.
    B = batch size, L = target sequence length
    Args:
        pred_ids: (B, L)
        tgt_ids: (B, L)
        eos_token_idx: int
    Returns:
        Tensor of one element
    """
    hit = (pred_ids == tgt_ids).long()
    eos = tgt_ids == eos_token_idx
    return (
        (hit.cumsum(dim=-1)[eos.roll(-1, dims=-1)] == eos.nonzero(as_tuple=True)[1])
        .float()
        .mean()
    )


def calc_sequence_acc_na(pred_ids: LongTensor, tgt_ids: LongTensor) -> Tensor:
    """
    Checks how many sequences in a batch are predicted perfectly.
    Intented for use with non-autoregressive predictions.
    B = batch size, L = maximum sequence length
    Args:
        pred_ids: (B, L)
        tgt_ids: (B, L)
    Returns:
        Tensor of one element
    """
    hit = pred_ids == tgt_ids
    return (hit.sum(-1) == hit.size(-1)).float().mean()
