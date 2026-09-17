import pytest
import torch
from PIL import Image

from core.data_loader import DeterministicWeightedSampler, _validate_paired_image


def test_weighted_sampler_resumes_from_generator_state():
    generator = torch.Generator().manual_seed(777)
    sampler = DeterministicWeightedSampler([1.0, 2.0, 3.0], 20, generator)
    iterator = iter(sampler)
    first = [next(iterator) for _ in range(4)]
    state = generator.get_state()
    expected_continuation = [next(iterator) for _ in range(5)]

    resumed_generator = torch.Generator()
    resumed_generator.set_state(state)
    resumed = DeterministicWeightedSampler([1.0, 2.0, 3.0], 20, resumed_generator)
    resumed_iterator = iter(resumed)

    assert len(first) == 4
    assert [next(resumed_iterator) for _ in range(5)] == expected_continuation


def test_paired_image_shape_validation():
    valid = Image.new("RGB", (512, 256))
    assert _validate_paired_image(valid) is valid

    with pytest.raises(ValueError, match=r"width=2\*height"):
        _validate_paired_image(Image.new("RGB", (256, 256)))
