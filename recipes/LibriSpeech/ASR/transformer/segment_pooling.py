"""Boundary-driven segment pooling utilities (policy-free).

Shared by the SpeechLLM recipes:

* ``train_speechllm.py`` — fixed-segmentation baselines (``boundary_source``):
  mean-pool every ``k`` frames, or mean-pool between externally provided
  alignment boundaries, with the downstream trained end-to-end.
* ``segmenter.py`` / ``train_speechllm_with_segmenter.py`` — the learned
  segmenter (CNN boundary policy + REINFORCE) uses the same aggregators, so
  baselines and the learned model differ *only* in where boundaries come from.

Conventions (documented on purpose — do not assume "0 = boundary"):
    boundary tensor values, shape ``(B, T)``:
        ``1``  = boundary  -> this frame *starts a new segment*
        ``0``  = continuation
        ``-1`` = padding
    ``k`` interior boundaries produce ``k + 1`` segments. Frame 0 always starts
    segment 0; a boundary predicted at frame 0 is a no-op (this is the frame-0
    guard — it prevents the divide-by-zero / phantom-segment bug).

    ``padding_mask`` (B, T) is a boolean tensor with ``True == padding``. The
    surrounding recipes use SpeechBrain *relative* lengths (floats in
    ``[0, 1]``); convert with ``lengths_to_padding_mask`` /
    ``padding_mask_to_lengths`` at the boundary.

"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Mask <-> relative-length helpers (bridge to the SpeechBrain recipe convention)
# ---------------------------------------------------------------------------


def lengths_to_padding_mask(rel_lens, max_len):
    """Convert SpeechBrain relative lengths to a boolean padding mask.

    Arguments
    ---------
    rel_lens : torch.Tensor
        Relative lengths, shape ``(B,)``, values in ``[0, 1]``.
    max_len : int
        The padded time dimension ``T``.

    Returns
    -------
    padding_mask : torch.Tensor
        Boolean tensor ``(B, T)`` with ``True`` at padding positions. The number
        of valid frames per row matches the recipe convention
        ``round(rel_len * T)`` (see ``get_multimodal_attention_mask``).
    """
    abs_lens = torch.round(rel_lens * max_len).long()
    idx = torch.arange(max_len, device=rel_lens.device).unsqueeze(0)
    return idx >= abs_lens.unsqueeze(1)


def padding_mask_to_lengths(padding_mask):
    """Convert a boolean padding mask back to relative lengths.

    Arguments
    ---------
    padding_mask : torch.Tensor
        Boolean tensor ``(B, T)`` with ``True`` at padding positions.

    Returns
    -------
    rel_lens : torch.Tensor
        Relative lengths ``(B,)`` in ``[0, 1]`` (``0`` if ``T == 0``).
    """
    max_len = padding_mask.size(1)
    valid = (~padding_mask).sum(dim=1).float()
    if max_len == 0:
        return torch.zeros_like(valid)
    return valid / max_len


# ---------------------------------------------------------------------------
# Boundary generation (non-learned sources)
# ---------------------------------------------------------------------------


def fixed_rate_boundary_targets(padding_mask, k):
    """Fixed-rate boundaries: a boundary every ``k`` frames.

    Used both as the fixed mean-pooling baseline (``boundary_source:
    fixed_rate``) and as supervised warm-start targets for the learned
    segmenter.

    Arguments
    ---------
    padding_mask : torch.Tensor
        ``(B, T)`` boolean, ``True`` at padding.
    k : int
        Segment length in frames (``k >= 1``). Boundaries fall at frames
        ``k, 2k, ...`` (frame 0 is never a boundary — it always starts seg 0).

    Returns
    -------
    targets : torch.Tensor
        ``(B, T)`` long with values ``{1, 0, -1}`` (pad == ``-1``).
    """
    B, T = padding_mask.shape
    frame_idx = torch.arange(T, device=padding_mask.device).unsqueeze(0)
    targets = ((frame_idx % k == 0) & (frame_idx > 0)).long().expand(B, T)
    targets = targets.masked_fill(padding_mask, -1)
    return targets.clone()


# ---------------------------------------------------------------------------
# Segment-id assignment (vectorized, frame-0 safe) — shared by all aggregators
# ---------------------------------------------------------------------------


def compute_segment_ids(boundary, padding_mask):
    """Assign each valid frame to a 0-indexed segment id, vectorized.

    Example:
    If
    boundary = [[0, 0, 1, 0, 1, 0]]
    padding_mask = [[False, False, False, False, False, True]]

    is_first_valid = [[1, 0, 0, 0, 0, 0]]
    starts_new = [[1, 0, 1, 0, 1, 0]]
    seg_ids = [[0, 0, 1, 1, 2, -1]]

    A new segment starts at frame 0 (always) and at any later frame with
    ``boundary == 1``. This is exactly the frame-0 guard: a boundary predicted
    at frame 0 does *not* open an empty preceding segment, so every segment is
    guaranteed to contain at least one frame (no ``0/0`` and no phantom segment).

    Arguments
    ---------
    boundary : torch.Tensor
        ``(B, T)`` with values in ``{1, 0, -1}``.
    padding_mask : torch.Tensor
        ``(B, T)`` boolean, ``True`` at padding.

    Returns
    -------
    seg_ids : torch.Tensor
        ``(B, T)`` long. Valid frames hold their 0-indexed segment id; padding
        frames hold ``-1``.
    num_segments : torch.Tensor
        ``(B,)`` long, number of segments per utterance (``0`` for empty rows).
    max_num_segments : int
        ``max(num_segments)`` (at least ``1`` so downstream tensors are non-empty).
    """
    B, T = boundary.shape
    valid = ~padding_mask
    device = boundary.device
    frame_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

    is_first_valid = frame_idx == 0  # frame 0 always starts a segment
    starts_new = ((boundary == 1) | is_first_valid) & valid
    # 0-indexed segment id = (#starts up to and including t) - 1
    seg_ids = torch.cumsum(starts_new.long(), dim=1) - 1
    seg_ids = seg_ids.masked_fill(~valid, -1)

    num_segments = starts_new.sum(dim=1).long()  # (B,)
    max_num_segments = int(num_segments.max().item()) if B > 0 else 1
    max_num_segments = max(max_num_segments, 1)
    return seg_ids, num_segments, max_num_segments


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------


def mean_pool_segments(features, padding_mask, boundary):
    """Mean-pool frames within each segment (vectorized via ``index_add_``).

    Arguments
    ---------
    features : torch.Tensor
        ``(B, T, C)`` input features.
    padding_mask : torch.Tensor
        ``(B, T)`` boolean, ``True`` at padding.
    boundary : torch.Tensor
        ``(B, T)`` boundary decisions ``{1, 0, -1}``.

    Returns
    -------
    new_features : torch.Tensor
        ``(B, T', C)`` segment embeddings (``T'`` == max #segments).
    new_padding_mask : torch.Tensor
        ``(B, T')`` boolean padding mask for the pooled sequence.
    """
    B, T, C = features.shape
    seg_ids, num_segments, T_new = compute_segment_ids(boundary, padding_mask)
    valid = ~padding_mask  # (B, T)

    # Global scatter index into a flat (B * T_new, C) buffer.
    batch_offset = torch.arange(B, device=features.device).unsqueeze(1) * T_new
    gidx = (seg_ids + batch_offset).clamp(min=0)  # padding rows clamped
    flat_gidx = gidx[valid]  # (num_valid,)
    flat_feats = features[valid]  # (num_valid, C)

    sums = features.new_zeros(B * T_new, C)
    sums.index_add_(0, flat_gidx, flat_feats)
    counts = features.new_zeros(B * T_new)
    counts.index_add_(
        0, flat_gidx, torch.ones_like(flat_gidx, dtype=features.dtype)
    )
    counts = counts.clamp(min=1.0)  # empty slots stay zero after divide
    new_features = (sums / counts.unsqueeze(1)).view(B, T_new, C)

    idx = torch.arange(T_new, device=features.device).unsqueeze(0)
    new_padding_mask = idx >= num_segments.unsqueeze(1)
    new_features = new_features.masked_fill(new_padding_mask.unsqueeze(-1), 0.0)
    return new_features, new_padding_mask


# ---------------------------------------------------------------------------
# Blank / separator-token treatments (post-pool) — for the blank-removal ablations
# ---------------------------------------------------------------------------


def segment_separator_mask(is_sep_frame, boundary, padding_mask):
    """Per-segment separator flag ``(B, T')`` from per-frame is-separator labels.

    Maps a per-frame ``{0,1}`` separator label (1 = frame belongs to a
    separator/"blank" segment; produced by
    ``ctc_boundary_align.word_ctc_separator_frames``) onto the pooled segment
    axis, so a caller can identify which pooled audio tokens are the
    separator/gap tokens. ``is_sep_frame`` is homogeneous within a segment by
    construction, so any frame of a segment determines the segment's flag.

    Arguments
    ---------
    is_sep_frame : torch.Tensor
        ``(B, T)`` per-frame separator labels (``{0,1}``), same frame axis and
        padding as ``boundary``.
    boundary : torch.Tensor
        ``(B, T)`` boundary decisions ``{1, 0, -1}`` (same as ``mean_pool_segments``).
    padding_mask : torch.Tensor
        ``(B, T)`` boolean, ``True`` at padding.

    Returns
    -------
    sep_seg : torch.Tensor
        ``(B, T')`` boolean, ``True`` where the pooled segment is a separator
        segment. ``T'`` matches ``mean_pool_segments``'s output length.
    """
    seg_ids, _, T_new = compute_segment_ids(boundary, padding_mask)
    B = boundary.size(0)
    valid = ~padding_mask
    sep_seg = torch.zeros(B, T_new, dtype=torch.bool, device=boundary.device)
    b_idx, t_idx = torch.nonzero(valid, as_tuple=True)
    # Duplicate (b, seg) writes all carry the same value (homogeneous per seg),
    # so plain index assignment is well-defined.
    sep_seg[b_idx, seg_ids[b_idx, t_idx]] = is_sep_frame[b_idx, t_idx].bool()
    return sep_seg


def apply_blank_mode(
    features, padding_mask, sep_seg, mode, const_vec=None, generator=None
):
    """Apply a separator/"blank"-token treatment to a pooled sequence.

    Used by the blank-removal ablations (``blank_mode`` in
    ``train_speechllm.py``). The word (non-separator) tokens are never modified;
    only the separator tokens are dropped or have their content replaced.

    Arguments
    ---------
    features : torch.Tensor
        ``(B, T', C)`` pooled segment embeddings.
    padding_mask : torch.Tensor
        ``(B, T')`` boolean, ``True`` at padding.
    sep_seg : torch.Tensor
        ``(B, T')`` boolean, ``True`` at separator segments (see
        :func:`segment_separator_mask`).
    mode : str
        ``keep`` (no-op) | ``drop`` (remove separator columns, left-pack the
        rest) | ``zero`` (separator embeddings -> 0) | ``const`` (separator
        embeddings -> ``const_vec``) | ``shuffle`` (permute separator embeddings
        within each utterance).
    const_vec : torch.Tensor or None
        ``(C,)`` replacement vector, required for ``mode='const'``.
    generator : torch.Generator or None
        Optional RNG for ``mode='shuffle'`` (for reproducible eval).

    Returns
    -------
    new_features : torch.Tensor
        ``(B, T'', C)`` (``T'' == T'`` for all modes except ``drop``).
    new_padding_mask : torch.Tensor
        ``(B, T'')`` boolean padding mask.
    """
    if mode == "keep":
        return features, padding_mask
    B, T_new, C = features.shape
    valid = ~padding_mask
    sep = sep_seg & valid  # only separator columns that are real segments

    if mode == "zero":
        return features.masked_fill(sep.unsqueeze(-1), 0.0), padding_mask
    if mode == "const":
        if const_vec is None:
            raise ValueError("blank_mode='const' requires const_vec of shape (C,).")
        cv = const_vec.to(features.dtype).view(1, 1, C).expand(B, T_new, C)
        return torch.where(sep.unsqueeze(-1), cv, features), padding_mask
    if mode == "shuffle":
        out = features.clone()
        for b in range(B):
            idx = torch.nonzero(sep[b]).flatten()
            if idx.numel() > 1:
                perm = idx[
                    torch.randperm(
                        idx.numel(), generator=generator, device=features.device
                    )
                ]
                out[b, idx] = features[b, perm]
        return out, padding_mask
    if mode in ("drop", "blank_only"):
        # drop: keep word tokens (discard separators, the co-worker's ablation).
        # blank_only: the inverse -- keep ONLY separator tokens (discard words),
        # to probe how much the gap tokens alone encode.
        keep = valid & (~sep_seg) if mode == "drop" else valid & sep_seg
        new_lens = keep.sum(dim=1)
        T_out = max(int(new_lens.max().item()) if B > 0 else 0, 1)
        new_features = features.new_zeros(B, T_out, C)
        new_padding_mask = torch.ones(
            B, T_out, dtype=torch.bool, device=features.device
        )
        # Left-pack kept columns: destination index = running count of kept-so-far.
        dest = torch.cumsum(keep.long(), dim=1) - 1  # (B, T')
        src_b, src_t = torch.nonzero(keep, as_tuple=True)
        dst_t = dest[src_b, src_t]
        new_features[src_b, dst_t] = features[src_b, src_t]
        new_padding_mask[src_b, dst_t] = False
        return new_features, new_padding_mask
    raise ValueError(
        f"Unknown blank_mode '{mode}'. "
        "Use keep|drop|zero|const|shuffle|blank_only."
    )


class MeanPoolAggregator(nn.Module):
    """Mean-pool frames within each segment. Output dim == input dim.

    Thin module wrapper around :func:`mean_pool_segments` (parameter-free),
    kept as an ``nn.Module`` so it is interchangeable with
    :class:`RecurrentAggregator` in the learned segmenter.

    Arguments
    ---------
    input_dim : int
        Feature dimension ``C``. ``out_dim`` is set equal to it.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.out_dim = input_dim

    def forward(self, features, padding_mask, boundary):
        """Aggregate. See :func:`mean_pool_segments` for the contract."""
        return mean_pool_segments(features, padding_mask, boundary)


class RecurrentAggregator(nn.Module):
    """Aggregate each segment with an LSTM, taking the last valid hidden state.

    Experimental alternative to mean-pooling. Segments are contiguous ranges of
    frames, so the within-segment position of a frame is ``t - start_of_segment``.
    All segments across the batch are packed into a single padded tensor and run
    through one LSTM call, then the final hidden state of each segment is scattered
    back into the pooled sequence.

    Arguments
    ---------
    input_dim : int
        Feature dimension ``C`` of the input frames.
    hidden_dim : int
        LSTM hidden size. Also the output dimension (doubled if bidirectional).
    num_layers : int
        Number of stacked LSTM layers.
    bidirectional : bool
        If ``True``, concatenate both directions' final states.
    """

    def __init__(
        self, input_dim, hidden_dim, num_layers=1, bidirectional=False
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim * self.num_directions

    def forward(self, features, padding_mask, boundary):
        """Aggregate. See :func:`mean_pool_segments` for the contract."""
        B, T, C = features.shape
        seg_ids, num_segments, T_new = compute_segment_ids(
            boundary, padding_mask
        )
        device = features.device
        valid = ~padding_mask

        # Absolute segment index across the whole batch: seg_offset[b] + seg_id.
        seg_offset = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=device),
                torch.cumsum(num_segments, dim=0)[:-1],
            ]
        )
        total_segments = int(num_segments.sum().item())
        if total_segments == 0:
            new_features = features.new_zeros(B, T_new, self.out_dim)
            new_padding_mask = torch.ones(
                B, T_new, dtype=torch.bool, device=device
            )
            return new_features, new_padding_mask

        frame_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        global_seg = (seg_ids + seg_offset.unsqueeze(1)).clamp(min=0)  # (B, T)

        # Segment start time = min frame index within a segment. Segments are
        # contiguous, so this is well-defined; compute via scatter-reduce amin.
        seg_start = torch.full(
            (total_segments,), T, dtype=torch.long, device=device
        )
        seg_start.scatter_reduce_(
            0, global_seg[valid], frame_idx[valid], reduce="amin"
        )
        seg_len = torch.zeros(total_segments, dtype=torch.long, device=device)
        seg_len.scatter_reduce_(
            0,
            global_seg[valid],
            torch.ones_like(global_seg[valid]),
            reduce="sum",
        )
        max_seg_len = int(seg_len.max().item())

        # Build (total_segments, max_seg_len, C) packed segment tensor.
        pos_in_seg = frame_idx - seg_start[global_seg]  # (B, T)
        packed = features.new_zeros(total_segments, max_seg_len, C)
        packed[global_seg[valid], pos_in_seg[valid]] = features[valid]

        lengths = seg_len.clamp(min=1).cpu()
        packed_seq = nn.utils.rnn.pack_padded_sequence(
            packed, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed_seq)
        # h_n: (num_layers * num_directions, total_segments, hidden)
        if self.num_directions == 2:
            last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            last = h_n[-1]  # (total_segments, out_dim)

        new_features = features.new_zeros(B, T_new, self.out_dim)
        seg_row = (
            torch.arange(B, device=device).unsqueeze(1).expand(B, T_new)
        )
        local_seg = torch.arange(T_new, device=device).unsqueeze(0).expand(
            B, T_new
        )
        seg_valid = local_seg < num_segments.unsqueeze(1)  # (B, T_new)
        abs_idx = (seg_offset.unsqueeze(1) + local_seg)[seg_valid]
        new_features[seg_valid] = last[abs_idx]

        new_padding_mask = ~seg_valid
        return new_features, new_padding_mask


class EndpointResidualAggregator(nn.Module):
    """Add an order-sensitive endpoint residual to the segment mean.

    The residual sees ``[first, last, last - first, log(1 + length)]``.  Its
    output layer is initialized to zero, so a newly enabled module is *exactly*
    mean pooling before its first update.  This makes it safe to place in front
    of a decoder initialized from a mean-pooled checkpoint.

    Unlike mean, max, or statistics pooling, exchanging the first and last
    frames changes the residual input.  This is the cheapest order-aware
    diagnostic in the prototype.
    """

    def __init__(self, input_dim, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.out_dim = input_dim
        self.hidden = nn.Linear(3 * input_dim + 1, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, input_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, padding_mask, boundary):
        mean, new_padding_mask = mean_pool_segments(
            features, padding_mask, boundary
        )
        B, T, C = features.shape
        seg_ids, _, T_new = compute_segment_ids(
            boundary, padding_mask
        )
        valid = ~padding_mask
        frame_idx = torch.arange(T, device=features.device).unsqueeze(0)
        starts = valid & (
            (frame_idx == 0) | ((boundary == 1) & (frame_idx > 0))
        )
        next_starts = torch.zeros_like(starts)
        if T > 1:
            next_starts[:, :-1] = starts[:, 1:]
        next_padding = torch.cat(
            [padding_mask[:, 1:], torch.ones_like(padding_mask[:, :1])],
            dim=1,
        )
        ends = valid & (next_starts | next_padding)

        batch_offset = (
            torch.arange(B, device=features.device).unsqueeze(1) * T_new
        )
        global_ids = (seg_ids + batch_offset).clamp(min=0)
        first = features.new_zeros(B * T_new, C)
        last = features.new_zeros(B * T_new, C)
        first[global_ids[starts]] = features[starts]
        last[global_ids[ends]] = features[ends]

        counts = features.new_zeros(B * T_new)
        counts.index_add_(
            0,
            global_ids[valid],
            torch.ones_like(global_ids[valid], dtype=features.dtype),
        )
        length_feature = torch.log1p(counts).unsqueeze(1)
        ordered = torch.cat([first, last, last - first, length_feature], dim=1)
        delta = self.output(
            self.dropout(torch.nn.functional.gelu(self.hidden(ordered)))
        )
        delta = delta.view(B, T_new, C)
        result = mean + delta
        result = result.masked_fill(new_padding_mask.unsqueeze(-1), 0.0)
        return result, new_padding_mask


class ResidualRecurrentAggregator(nn.Module):
    """Encode each segment with a BiGRU/BiLSTM and add it to the mean.

    All real segments in a minibatch are packed into one recurrent call.  The
    final forward/backward states are projected back to ``input_dim`` and added
    to the ordinary segment mean.  The projection is zero-initialized, giving
    an exact mean-pooling warm start while retaining an order-aware learning
    path.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=1,
        dropout=0.0,
        cell="gru",
    ):
        super().__init__()
        cell = cell.lower()
        if cell not in ("gru", "lstm"):
            raise ValueError(f"Unknown recurrent cell {cell!r}; use 'gru' or 'lstm'.")
        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.cell = cell
        self.out_dim = input_dim
        self.output = nn.Linear(2 * hidden_dim, input_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, padding_mask, boundary):
        mean, new_padding_mask = mean_pool_segments(
            features, padding_mask, boundary
        )
        B, T, C = features.shape
        seg_ids, num_segments, T_new = compute_segment_ids(
            boundary, padding_mask
        )
        device = features.device
        valid = ~padding_mask
        total_segments = int(num_segments.sum().item())
        if total_segments == 0:
            return mean, new_padding_mask

        seg_offset = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=device),
                torch.cumsum(num_segments, dim=0)[:-1],
            ]
        )
        frame_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        global_seg = (seg_ids + seg_offset.unsqueeze(1)).clamp(min=0)
        seg_start = torch.full(
            (total_segments,), T, dtype=torch.long, device=device
        )
        seg_start.scatter_reduce_(
            0, global_seg[valid], frame_idx[valid], reduce="amin"
        )
        seg_len = torch.zeros(total_segments, dtype=torch.long, device=device)
        seg_len.scatter_reduce_(
            0,
            global_seg[valid],
            torch.ones_like(global_seg[valid]),
            reduce="sum",
        )
        max_seg_len = int(seg_len.max().item())
        pos_in_seg = frame_idx - seg_start[global_seg]
        packed = features.new_zeros(total_segments, max_seg_len, C)
        packed[global_seg[valid], pos_in_seg[valid]] = features[valid]
        packed_sequence = nn.utils.rnn.pack_padded_sequence(
            packed,
            seg_len.clamp(min=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, state = self.rnn(packed_sequence)
        hidden = state[0] if self.cell == "lstm" else state
        ordered = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        delta = self.output(ordered)

        result = mean.clone()
        rows = torch.arange(B, device=device).unsqueeze(1).expand(B, T_new)
        local_seg = torch.arange(T_new, device=device).unsqueeze(0).expand(
            B, T_new
        )
        segment_valid = local_seg < num_segments.unsqueeze(1)
        absolute_seg = (seg_offset.unsqueeze(1) + local_seg)[segment_valid]
        result[rows[segment_valid], local_seg[segment_valid]] += delta[
            absolute_seg
        ]
        result = result.masked_fill(new_padding_mask.unsqueeze(-1), 0.0)
        return result, new_padding_mask


def build_aggregator(
    name,
    input_dim,
    hidden_dim=128,
    num_layers=1,
    dropout=0.0,
    **kwargs,
):
    """Build a pooling module; ``mean`` remains the default production path."""
    name = (name or "mean").lower()
    if name in ("mean", "meanpool", "mean_pool"):
        return MeanPoolAggregator(input_dim)
    if name in ("lstm", "recurrent", "rnn"):
        return RecurrentAggregator(
            input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bidirectional=bool(kwargs.get("bidirectional", False)),
        )
    if name in ("endpoint_residual", "endpoint-residual", "endpoint"):
        return EndpointResidualAggregator(
            input_dim, hidden_dim=hidden_dim, dropout=dropout
        )
    if name in ("bigru_residual", "bigru-residual", "bigru"):
        return ResidualRecurrentAggregator(
            input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            cell="gru",
        )
    if name in ("bilstm_residual", "bilstm-residual", "bilstm"):
        return ResidualRecurrentAggregator(
            input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            cell="lstm",
        )
    raise ValueError(
        f"Unknown aggregator '{name}'. Use mean|endpoint_residual|"
        "bigru_residual|bilstm_residual (or legacy lstm)."
    )


if __name__ == "__main__":
    # Self-tests for the boundary/pooling machinery (policy-free).
    torch.manual_seed(0)
    B, T, C = 3, 20, 16
    feats = torch.randn(B, T, C)
    rel_lens = torch.tensor([1.0, 0.75, 0.5])
    pad = lengths_to_padding_mask(rel_lens, T)

    # Identity mode: all-ones boundaries -> every frame is its own segment.
    all_ones = torch.ones(B, T, dtype=torch.long).masked_fill(pad, -1)
    out, out_pad = mean_pool_segments(feats, pad, all_ones)
    for b in range(B):
        n = int((~pad[b]).sum())
        assert torch.allclose(out[b, :n], feats[b, :n], atol=1e-5), (
            "identity fail"
        )
    print("identity-mode mean-pool OK:", feats.shape, "->", out.shape)

    # Frame-0 boundary must NOT create a phantom/NaN segment.
    boundary = torch.zeros(B, T, dtype=torch.long)
    boundary[:, 0] = 1
    boundary = boundary.masked_fill(pad, -1)
    out, out_pad = mean_pool_segments(feats, pad, boundary)
    assert not torch.isnan(out).any(), "frame-0 guard failed (NaN)"
    _, num_seg, _ = compute_segment_ids(boundary, pad)
    assert (num_seg == 1).all(), f"frame-0 phantom segment: {num_seg}"
    print("frame-0 guard OK")

    # Fixed-rate boundaries: k=5 over 20 valid frames -> 4 segments, and the
    # pooled lengths round-trip exactly through the relative-length convention.
    fr = fixed_rate_boundary_targets(pad, 5)
    out, out_pad = mean_pool_segments(feats, pad, fr)
    _, num_seg, _ = compute_segment_ids(fr, pad)
    assert num_seg.tolist() == [4, 3, 2], f"fixed-rate segments: {num_seg}"
    new_lens = padding_mask_to_lengths(out_pad)
    assert torch.equal(
        torch.round(new_lens * out.size(1)).long(), num_seg
    ), "length round-trip fail"
    # First segment of row 0 == mean of frames 0..4.
    assert torch.allclose(out[0, 0], feats[0, :5].mean(0), atol=1e-5)
    print("fixed-rate pooling OK:", num_seg.tolist(), "segments")

    # LSTM aggregator smoke test.
    agg = RecurrentAggregator(C, hidden_dim=C)
    o, op = agg(feats, pad, fr)
    print("lstm aggregator OK:", o.shape, "out_dim", agg.out_dim)

    # Order-aware residual poolers must be checkpoint-safe: all are exactly mean
    # pooling at initialization, including their padding. Once the residual head
    # is nonzero, reversing a two-frame segment must change its representation.
    pair = torch.randn(1, 2, C)
    reversed_pair = pair.flip(1)
    pair_pad = torch.zeros(1, 2, dtype=torch.bool)
    pair_boundary = torch.zeros(1, 2, dtype=torch.long)
    pair_mean, pair_mean_pad = mean_pool_segments(
        pair, pair_pad, pair_boundary
    )
    for pool_name in (
        "endpoint_residual",
        "bigru_residual",
        "bilstm_residual",
    ):
        pooler = build_aggregator(pool_name, C, hidden_dim=8)
        initial, initial_pad = pooler(pair, pair_pad, pair_boundary)
        assert torch.equal(initial_pad, pair_mean_pad)
        assert torch.equal(initial, pair_mean), (
            f"{pool_name} must initialize to exact mean pooling"
        )
        with torch.no_grad():
            pooler.output.weight.normal_(std=0.1)
        forward, _ = pooler(pair, pair_pad, pair_boundary)
        reverse, _ = pooler(reversed_pair, pair_pad, pair_boundary)
        assert not torch.allclose(forward, reverse), (
            f"{pool_name} failed the a->e versus e->a order test"
        )
        train_pair = pair.clone().requires_grad_(True)
        train_out, _ = pooler(train_pair, pair_pad, pair_boundary)
        train_out.square().sum().backward()
        assert train_pair.grad is not None and torch.isfinite(
            train_pair.grad
        ).all()
        params = sum(p.numel() for p in pooler.parameters())
        print(
            f"{pool_name} OK: exact-mean init; reversal-sensitive after update; "
            f"gradients finite; {params:,} parameters at C={C}"
        )

    # Blank/separator treatments. 1 utt, 8 frames -> 4 segments of 2 frames;
    # segments 0 and 2 are separators (is_sep alternates like word_ctc).
    bf = torch.randn(1, 8, 4)
    bpad = torch.zeros(1, 8, dtype=torch.bool)
    bbound = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0]]).long()
    is_sep = torch.tensor([[1, 1, 0, 0, 1, 1, 0, 0]]).long()
    pooled, ppad = mean_pool_segments(bf, bpad, bbound)  # (1, 4, 4)
    sep_seg = segment_separator_mask(is_sep, bbound, bpad)
    assert sep_seg.tolist() == [[True, False, True, False]], sep_seg
    # keep == identity
    k, kp = apply_blank_mode(pooled, ppad, sep_seg, "keep")
    assert torch.equal(k, pooled)
    # drop: keep only word segments (1, 3), left-packed, length 2
    d, dp = apply_blank_mode(pooled, ppad, sep_seg, "drop")
    assert d.shape == (1, 2, 4), d.shape
    assert torch.allclose(d[0, 0], pooled[0, 1]) and torch.allclose(d[0, 1], pooled[0, 3])
    assert dp.tolist() == [[False, False]], dp
    # zero: separator rows zeroed, word rows untouched
    z, _ = apply_blank_mode(pooled, ppad, sep_seg, "zero")
    assert torch.allclose(z[0, 0], torch.zeros(4)) and torch.allclose(z[0, 2], torch.zeros(4))
    assert torch.allclose(z[0, 1], pooled[0, 1]) and torch.allclose(z[0, 3], pooled[0, 3])
    # const: separator rows == const_vec, word rows untouched
    cv = torch.arange(4).float()
    c, _ = apply_blank_mode(pooled, ppad, sep_seg, "const", const_vec=cv)
    assert torch.allclose(c[0, 0], cv) and torch.allclose(c[0, 2], cv)
    assert torch.allclose(c[0, 1], pooled[0, 1])
    # shuffle: word rows untouched; separator rows are a permutation of originals
    sh, _ = apply_blank_mode(pooled, ppad, sep_seg, "shuffle")
    assert torch.allclose(sh[0, 1], pooled[0, 1]) and torch.allclose(sh[0, 3], pooled[0, 3])
    sep_orig = torch.stack([pooled[0, 0], pooled[0, 2]])
    sep_new = torch.stack([sh[0, 0], sh[0, 2]])
    assert torch.allclose(sep_orig.sum(0), sep_new.sum(0)), "shuffle changed multiset"
    # blank_only: inverse of drop -- keep only separator segments (0, 2)
    bo, bop = apply_blank_mode(pooled, ppad, sep_seg, "blank_only")
    assert bo.shape == (1, 2, 4), bo.shape
    assert torch.allclose(bo[0, 0], pooled[0, 0]) and torch.allclose(bo[0, 1], pooled[0, 2])
    assert bop.tolist() == [[False, False]], bop
    print("blank-mode ops OK: keep/drop/zero/const/shuffle/blank_only")
