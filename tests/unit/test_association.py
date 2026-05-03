"""Associação surfista ↔ prancha."""

from surf_app.association import bbox_center, pair_person_surfboard


def test_bbox_center():
    assert bbox_center((0.0, 0.0, 10.0, 20.0)) == (5.0, 10.0)


def test_pair_prefers_high_iou():
    persons = [{'bbox': (0.0, 0.0, 10.0, 10.0), 'conf': 0.9}]
    boards = [
        {'bbox': (100.0, 100.0, 110.0, 110.0), 'conf': 0.8},
        {'bbox': (2.0, 2.0, 9.0, 9.0), 'conf': 0.85},
    ]
    pairs = pair_person_surfboard(persons, boards, frame_diag=500.0)
    assert len(pairs) == 1
    assert pairs[0][1] is boards[1]
