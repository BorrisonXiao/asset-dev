"""Correctness tests for segment_pooling.py and the fixed-boundary path of
train_speechllm.py.

Run with: python test_segment_pooling.py

Test inputs are written as human-readable per-frame segment labels, e.g.
``[1, 1, 1, 2, 2, 3, 3, 3]`` means frames 0-2 form segment 1, frames 3-4 form
segment 2, frames 5-7 form segment 3. Features are frame indices so expected
segment means can be verified by hand.

Failure cases covered (one test each):
 1. Known-answer segmentation: labels -> boundaries -> pooled means, batched
    with per-row different lengths AND different boundary patterns.
 2. Boundary at frame 0 must not create an empty "phantom" segment (0/0 NaN).
 3. All-zero boundaries -> exactly one segment == mean of all valid frames.
 4. All-one boundaries -> identity (every frame its own segment).
 5. Boundary=1 placed inside the padding region must be ignored.
 6. Fully-padded (empty) row -> 0 segments, zero output, all-pad mask, no NaN.
 7. fixed_rate_boundary_targets: correct boundary positions with and without
    padding; k=1 -> identity; k >= valid length -> single segment.
 8. Length round-trip: pooled mask -> relative lengths -> round() recovers the
    exact per-row segment count (the recipe's rounding convention).
 9. Integration with get_multimodal_attention_mask: the mask marks exactly
    num_segments audio positions per row.
10. ASR.get_fixed_boundaries: fixed_rate matches the helper; alignment applies
    the pad mask; alignment with mismatched frame count raises; unknown source
    raises.
11. RecurrentAggregator: same segmentation gives the same output padding as
    mean-pool, no NaN; all-empty batch returns zeros without crashing.
12. bf16 features (recipe precision) pool without error and match fp32.
"""

import types

import torch

from segment_pooling import (
    RecurrentAggregator,
    compute_segment_ids,
    fixed_rate_boundary_targets,
    lengths_to_padding_mask,
    mean_pool_segments,
    padding_mask_to_lengths,
)


def boundaries_from_labels(labels, T, pad_value=-1):
    """Human-readable per-frame segment labels -> (boundary, padding_mask) rows.

    ``labels`` is a list like [1, 1, 1, 2, 2, 3, 3, 3]; frames beyond
    len(labels) are padding. Returns 1-row tensors of length T.
    """
    n = len(labels)
    boundary = torch.zeros(1, T, dtype=torch.long)
    for t in range(1, n):
        if labels[t] != labels[t - 1]:
            boundary[0, t] = 1
    pad = torch.ones(1, T, dtype=torch.bool)
    pad[0, :n] = False
    boundary = boundary.masked_fill(pad, pad_value)
    return boundary, pad


def frame_index_features(B, T, C=2):
    """features[b, t] = (t, 10*t) so segment means are hand-checkable."""
    idx = torch.arange(T, dtype=torch.float32)
    feats = torch.stack([idx, 10 * idx], dim=-1)  # (T, 2)
    return feats.unsqueeze(0).expand(B, T, C).clone()


def test_known_answer_batched():
    """Case 1: three rows with different lengths and boundary patterns."""
    T = 8
    rows = [
        # labels                          expected means (first channel)
        ([1, 1, 1, 2, 2, 3, 3, 3], [1.0, 3.5, 6.0]),
        ([1, 1, 2, 2, 3, 3], [0.5, 2.5, 4.5]),
        ([1, 2, 3, 4], [0.0, 1.0, 2.0, 3.0]),
    ]
    boundary = torch.cat(
        [boundaries_from_labels(labels, T)[0] for labels, _ in rows]
    )
    pad = torch.cat(
        [boundaries_from_labels(labels, T)[1] for labels, _ in rows]
    )
    feats = frame_index_features(len(rows), T)

    seg_ids, num_seg, T_new = compute_segment_ids(boundary, pad)
    # seg_ids must reproduce the 0-indexed labels, -1 at padding.
    for b, (labels, _) in enumerate(rows):
        expected = [x - 1 for x in labels] + [-1] * (T - len(labels))
        assert seg_ids[b].tolist() == expected, (b, seg_ids[b].tolist())
    assert num_seg.tolist() == [3, 3, 4]
    assert T_new == 4

    pooled, seg_pad = mean_pool_segments(feats, pad, boundary)
    assert pooled.shape == (3, 4, 2)
    for b, (_, means) in enumerate(rows):
        got = pooled[b, : len(means), 0].tolist()
        assert got == means, (b, got, means)
        # second channel is 10x the first
        assert pooled[b, : len(means), 1].tolist() == [10 * m for m in means]
        # slots past the last segment: padded and zeroed
        assert seg_pad[b].tolist() == [False] * len(means) + [True] * (
            4 - len(means)
        )
        assert (pooled[b, len(means):] == 0).all()
    print("1. known-answer batched pooling OK")


def test_frame0_boundary_guard():
    """Case 2: boundary at frame 0 is a no-op (frame 0 always starts seg 0)."""
    T = 6
    boundary, pad = boundaries_from_labels([1, 1, 1, 1], T)
    boundary[0, 0] = 1  # force a boundary on frame 0
    feats = frame_index_features(1, T)
    pooled, seg_pad = mean_pool_segments(feats, pad, boundary)
    _, num_seg, _ = compute_segment_ids(boundary, pad)
    assert num_seg.tolist() == [1], num_seg.tolist()
    assert not torch.isnan(pooled).any()
    assert pooled[0, 0, 0].item() == 1.5  # mean(0,1,2,3)
    print("2. frame-0 guard OK")


def test_all_zero_boundaries():
    """Case 3: no boundaries -> one segment == mean of valid frames only."""
    T = 8
    boundary, pad = boundaries_from_labels([1, 1, 1, 1, 1], T)
    feats = frame_index_features(1, T)
    pooled, seg_pad = mean_pool_segments(feats, pad, boundary)
    _, num_seg, _ = compute_segment_ids(boundary, pad)
    assert num_seg.tolist() == [1]
    assert pooled[0, 0, 0].item() == 2.0  # mean(0..4), padding excluded
    print("3. all-zero boundaries OK")


def test_all_one_boundaries_identity():
    """Case 4: boundary on every frame -> pooling is the identity."""
    T = 8
    valid_n = 5
    boundary, pad = boundaries_from_labels(list(range(1, valid_n + 1)), T)
    assert (boundary[0, 1:valid_n] == 1).all()  # sanity: all interior frames
    feats = torch.randn(1, T, 4)
    pooled, seg_pad = mean_pool_segments(feats, pad, boundary)
    assert torch.allclose(pooled[0, :valid_n], feats[0, :valid_n], atol=1e-6)
    assert (~seg_pad).sum() == valid_n
    print("4. all-one boundaries identity OK")


def test_boundary_in_padding_ignored():
    """Case 5: a stray boundary=1 inside padding must not create segments."""
    T = 8
    boundary, pad = boundaries_from_labels([1, 1, 2, 2], T)
    stray = boundary.clone()
    stray[0, 5] = 1  # padding position (only frames 0-3 are valid)
    stray[0, 7] = 1
    feats = frame_index_features(1, T)
    ref_pooled, ref_pad = mean_pool_segments(feats, pad, boundary)
    got_pooled, got_pad = mean_pool_segments(feats, pad, stray)
    assert torch.equal(ref_pooled, got_pooled)
    assert torch.equal(ref_pad, got_pad)
    _, num_seg, _ = compute_segment_ids(stray, pad)
    assert num_seg.tolist() == [2]
    print("5. boundary-in-padding ignored OK")


def test_empty_row():
    """Case 6: fully padded row -> 0 segments, zero features, no NaN."""
    T = 6
    boundary1, pad1 = boundaries_from_labels([1, 1, 2], T)
    pad_empty = torch.ones(1, T, dtype=torch.bool)
    boundary_empty = torch.full((1, T), -1, dtype=torch.long)
    boundary = torch.cat([boundary1, boundary_empty])
    pad = torch.cat([pad1, pad_empty])
    feats = frame_index_features(2, T)

    pooled, seg_pad = mean_pool_segments(feats, pad, boundary)
    _, num_seg, _ = compute_segment_ids(boundary, pad)
    assert num_seg.tolist() == [2, 0]
    assert seg_pad[1].all()  # empty row: everything padded
    assert (pooled[1] == 0).all()
    assert not torch.isnan(pooled).any()
    lens = padding_mask_to_lengths(seg_pad)
    assert lens[1].item() == 0.0
    print("6. empty-row OK")


def test_fixed_rate_targets():
    """Case 7: fixed-rate boundary positions, k=1 identity, k >= len."""
    T = 8
    # Full-length row, k=3: boundaries at frames 3 and 6 ->
    # labels [1,1,1,2,2,2,3,3]
    pad_full = torch.zeros(1, T, dtype=torch.bool)
    fr = fixed_rate_boundary_targets(pad_full, 3)
    assert fr.tolist() == [[0, 0, 0, 1, 0, 0, 1, 0]], fr.tolist()
    seg_ids, num_seg, _ = compute_segment_ids(fr, pad_full)
    assert seg_ids.tolist() == [[0, 0, 0, 1, 1, 1, 2, 2]]
    assert num_seg.tolist() == [3]

    # Padded row (4 valid), k=3: only frame 3 is a boundary.
    pad4 = torch.tensor([[False] * 4 + [True] * 4])
    fr = fixed_rate_boundary_targets(pad4, 3)
    assert fr.tolist() == [[0, 0, 0, 1, -1, -1, -1, -1]]
    _, num_seg, _ = compute_segment_ids(fr, pad4)
    assert num_seg.tolist() == [2]

    # k=1 -> every frame its own segment (identity).
    feats = torch.randn(1, T, 3)
    fr = fixed_rate_boundary_targets(pad4, 1)
    pooled, seg_pad = mean_pool_segments(feats, pad4, fr)
    assert torch.allclose(pooled[0, :4], feats[0, :4], atol=1e-6)

    # k >= valid length -> a single segment.
    fr = fixed_rate_boundary_targets(pad4, 100)
    _, num_seg, _ = compute_segment_ids(fr, pad4)
    assert num_seg.tolist() == [1]
    print("7. fixed-rate targets OK")


def test_length_round_trip():
    """Case 8: relative lengths recover exact segment counts after round()."""
    torch.manual_seed(0)
    B, T = 5, 53
    rel_lens = torch.tensor([1.0, 0.9, 0.62, 0.31, 0.07])
    pad = lengths_to_padding_mask(rel_lens, T)
    feats = torch.randn(B, T, 4)
    for k in (1, 2, 4, 5, 7, 50):
        fr = fixed_rate_boundary_targets(pad, k)
        pooled, seg_pad = mean_pool_segments(feats, pad, fr)
        _, num_seg, _ = compute_segment_ids(fr, pad)
        seg_lens = padding_mask_to_lengths(seg_pad)
        recovered = torch.round(seg_lens * pooled.size(1)).long()
        assert torch.equal(recovered, num_seg), (k, recovered, num_seg)
    print("8. length round-trip OK for k in {1,2,4,5,7,50}")


def test_attention_mask_integration():
    """Case 9: recipe attention mask marks exactly num_segments audio slots."""
    from train_speechllm import get_multimodal_attention_mask

    torch.manual_seed(0)
    B, T, C = 4, 53, 8
    rel_lens = torch.tensor([1.0, 0.9, 0.62, 0.31])
    pad = lengths_to_padding_mask(rel_lens, T)
    feats = torch.randn(B, T, C)
    fr = fixed_rate_boundary_targets(pad, 5)
    pooled, seg_pad = mean_pool_segments(feats, pad, fr)
    seg_lens = padding_mask_to_lengths(seg_pad)
    _, num_seg, _ = compute_segment_ids(fr, pad)

    txt = torch.randn(B, 11, C)
    txt_lens = torch.tensor([1.0, 0.8, 0.6, 0.4])
    mask = get_multimodal_attention_mask(pooled, seg_lens, txt, txt_lens, "cpu")
    # Layout: [start_of_audio][audio slots][remaining text]
    for i in range(B):
        audio_true = int(mask[i, 1 : 1 + pooled.size(1)].sum())
        assert audio_true == int(num_seg[i]), (i, audio_true, int(num_seg[i]))
    print("9. attention-mask integration OK:", num_seg.tolist())


def test_get_fixed_boundaries():
    """Case 10: the recipe-side boundary dispatcher, incl. error paths."""
    from train_speechllm import ASR

    T = 8
    pad = torch.tensor([[False] * 6 + [True] * 2, [False] * 4 + [True] * 4])
    fake_self = types.SimpleNamespace(
        hparams=types.SimpleNamespace(fixed_rate_k=3)
    )

    # fixed_rate delegates to the helper.
    got = ASR.get_fixed_boundaries(fake_self, "fixed_rate", pad, None)
    assert torch.equal(got, fixed_rate_boundary_targets(pad, 3))

    # alignment: applies the pad mask (stray labels in padding -> -1).
    bt = torch.tensor(
        [[0, 0, 1, 0, 1, 0, 1, 1], [0, 1, 0, 1, 0, 0, 0, 0]]
    )
    batch = types.SimpleNamespace(
        boundary_target=(bt, torch.tensor([0.75, 0.5]))
    )
    got = ASR.get_fixed_boundaries(fake_self, "alignment", pad, batch)
    assert got.tolist() == [
        [0, 0, 1, 0, 1, 0, -1, -1],
        [0, 1, 0, 1, -1, -1, -1, -1],
    ]

    # alignment with mismatched frame count must raise.
    bad = types.SimpleNamespace(
        boundary_target=(bt[:, :5], torch.tensor([1.0, 0.8]))
    )
    try:
        ASR.get_fixed_boundaries(fake_self, "alignment", pad, bad)
        raise AssertionError("expected ValueError for frame-count mismatch")
    except ValueError as e:
        assert "frames" in str(e)

    # unknown source must raise.
    try:
        ASR.get_fixed_boundaries(fake_self, "bogus", pad, None)
        raise AssertionError("expected ValueError for unknown source")
    except ValueError:
        pass
    print("10. get_fixed_boundaries OK (incl. error paths)")


def test_recurrent_aggregator():
    """Case 11: LSTM aggregator agrees on shapes/padding; empty batch safe."""
    torch.manual_seed(0)
    T = 8
    boundary, pad = boundaries_from_labels([1, 1, 1, 2, 2, 3, 3, 3], T)
    boundary2, pad2 = boundaries_from_labels([1, 2, 2], T)
    boundary = torch.cat([boundary, boundary2])
    pad = torch.cat([pad, pad2])
    feats = torch.randn(2, T, 4)

    agg = RecurrentAggregator(4, hidden_dim=6)
    out, out_pad = agg(feats, pad, boundary)
    ref_out, ref_pad = mean_pool_segments(feats, pad, boundary)
    assert out.shape == (2, ref_out.size(1), 6)
    assert torch.equal(out_pad, ref_pad)
    assert not torch.isnan(out).any()

    # All rows empty: must return zeros without crashing (total_segments == 0).
    pad_all = torch.ones(2, T, dtype=torch.bool)
    boundary_all = torch.full((2, T), -1, dtype=torch.long)
    out, out_pad = agg(feats, pad_all, boundary_all)
    assert (out == 0).all() and out_pad.all()
    print("11. recurrent aggregator OK")


def test_bf16():
    """Case 12: bf16 features (recipe precision) pool correctly."""
    torch.manual_seed(0)
    T = 8
    boundary, pad = boundaries_from_labels([1, 1, 1, 2, 2, 3, 3, 3], T)
    feats = torch.randn(1, T, 4)
    ref, _ = mean_pool_segments(feats, pad, boundary)
    got, _ = mean_pool_segments(feats.to(torch.bfloat16), pad, boundary)
    assert got.dtype == torch.bfloat16
    assert torch.allclose(ref, got.float(), atol=0.05)
    print("12. bf16 pooling OK")


if __name__ == "__main__":
    test_known_answer_batched()
    test_frame0_boundary_guard()
    test_all_zero_boundaries()
    test_all_one_boundaries_identity()
    test_boundary_in_padding_ignored()
    test_empty_row()
    test_fixed_rate_targets()
    test_length_round_trip()
    test_attention_mask_integration()
    test_get_fixed_boundaries()
    test_recurrent_aggregator()
    test_bf16()
    print("\nall segment-pooling tests passed")
