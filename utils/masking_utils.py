
import torch


# create pad mask request x [ bs, seq_len, d_model ] -> mask shape [ bs, num_rows, seq_len ]
def create_pad_mask(mask_size, pad_along_row_input, pad_along_column_input, rank):
    batch_size, output_seq_len, input_seq_len = mask_size
    mask = torch.ones(size=(batch_size, output_seq_len, input_seq_len), dtype=torch.int8).to(rank)
    for batch_idx in range(batch_size):
        mask[batch_idx, :, (input_seq_len - pad_along_column_input[batch_idx]):] = 0
        mask[batch_idx, (output_seq_len - pad_along_row_input[batch_idx]):, :] = 0
    # output [ batch_size, output_seq_len, input_seq_len ]
    return mask


# create_no_peak_and_pad_mask create given x shape [bs, seq_len, d_model] -> mask shape [bs, seq_len, seq_len] for
# the self attention modules
def create_no_peak_and_pad_mask(mask_size, num_pads, rank):
    batch_size, seq_len, seq_len = mask_size

    mask = torch.tril(torch.ones(size=(seq_len, seq_len), dtype=torch.int8),
                      diagonal=0).unsqueeze(0).repeat(batch_size, 1, 1).to(rank)
    for batch_idx in range(batch_size):
        mask[batch_idx, :, seq_len - num_pads[batch_idx]:] = 0
        mask[batch_idx, (seq_len - num_pads[batch_idx]):, :] = 0
    return mask
