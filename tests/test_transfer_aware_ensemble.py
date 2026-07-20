"""CPU checks for transfer-aware LR-RGDA ensemble routing.

Run on the training environment with:
    python tests/test_transfer_aware_ensemble.py
"""

import torch
from torchvision.transforms import Resize

from src.utils.data import get_transforms
from src.utils.main_utils import combine_ensemble_logits


def test_zs_predicted_seen_routing():
    # Classes 0-1 are already learned; 2-3 are future/unseen classes.
    # Row 0 is classified as unseen by ZS, row 1 as seen by ZS.
    zs_logits = torch.tensor([
        [1.0, 0.0, 4.0, 3.0],
        [4.0, 1.0, 3.0, 2.0],
    ])
    rgda_logits = torch.tensor([
        [0.0, -3.0],
        [-3.0, 0.0],
    ])

    classwise = combine_ensemble_logits(
        zs_logits, rgda_logits, 2, 0.5, mode="raw", routing="classwise"
    )
    gated = combine_ensemble_logits(
        zs_logits, rgda_logits, 2, 0.5, mode="raw",
        routing="zs_predicted_seen",
    )

    # An unseen ZS prediction is preserved exactly; a seen ZS prediction uses
    # the existing classwise ensemble unchanged.
    torch.testing.assert_close(gated[0], zs_logits[0])
    torch.testing.assert_close(gated[1], classwise[1])
    assert gated[0].argmax().item() == zs_logits[0].argmax().item()


def test_evaluation_resize_modes():
    _, legacy_test = get_transforms("aircraft", test_resize_mode="legacy_square")
    _, aspect_test = get_transforms("aircraft", test_resize_mode="preserve_aspect")
    legacy_resize = next(t for t in legacy_test.transforms if isinstance(t, Resize))
    aspect_resize = next(t for t in aspect_test.transforms if isinstance(t, Resize))

    assert legacy_resize.size == (224, 224)
    assert aspect_resize.size == 224


if __name__ == "__main__":
    test_zs_predicted_seen_routing()
    test_evaluation_resize_modes()
    print("transfer-aware ensemble: passed")
