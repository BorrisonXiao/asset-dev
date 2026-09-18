"""Tests for the corrected multimodal decoding layout."""

import torch

from recipes.LibriSpeech.ASR.transformer.batch_invariant_decode import (
    compact_position_ids,
    duration_decode_limits,
    left_pack_valid_prefix,
)


def test_compact_positions_ignore_interior_and_left_padding():
    interior = torch.tensor([[1, 1, 0, 0, 1, 1]], dtype=torch.bool)
    left = torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.bool)
    assert compact_position_ids(interior).tolist() == [[0, 1, 0, 0, 2, 3]]
    assert compact_position_ids(left).tolist() == [[0, 0, 0, 1, 2, 3]]
    assert compact_position_ids(interior)[interior].tolist() == [0, 1, 2, 3]
    assert compact_position_ids(left)[left].tolist() == [0, 1, 2, 3]


def test_left_pack_removes_interior_padding_without_reordering():
    states = torch.tensor(
        [
            [[1.0], [2.0], [0.0], [0.0], [3.0], [4.0]],
            [[5.0], [6.0], [7.0], [8.0], [9.0], [10.0]],
        ]
    )
    mask = torch.tensor(
        [[1, 1, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1]], dtype=torch.bool
    )
    packed, packed_mask = left_pack_valid_prefix(states, mask)
    assert packed[:, :, 0].tolist() == [
        [0.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    ]
    assert packed_mask.tolist() == [
        [False, False, True, True, True, True],
        [True, True, True, True, True, True],
    ]


def test_duration_limits_do_not_depend_on_batchmate_or_audio_token_rate():
    alone = duration_decode_limits(torch.tensor([7.25]), 12.0, 16, 512)
    mixed = duration_decode_limits(
        torch.tensor([7.25, 21.0]), 12.0, 16, 512
    )
    assert alone.item() == mixed[0].item() == 103
    assert mixed[1].item() == 268

