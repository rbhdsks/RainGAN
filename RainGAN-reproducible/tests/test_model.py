from types import SimpleNamespace

import torch
from munch import Munch
from PIL import Image

from core.data_loader import get_test_loader
from core.model import build_model
from core.solver import Solver, mutual_info


def test_model_forward_shapes_on_cpu():
    args = SimpleNamespace(
        img_size=64,
        style_dim=16,
        latent_dim=8,
        num_domains=2,
        w_hpf=0,
    )
    networks, _ = build_model(args)
    image = torch.randn(1, 3, 64, 64)
    label = torch.ones(1, dtype=torch.long)
    latent = torch.randn(1, 8)
    style = networks.mapping_network(latent, label)
    generated = networks.generator(image, style)

    assert style.shape == (1, 16)
    assert generated.shape == image.shape
    assert networks.discriminator(generated, label).shape == (1,)


def test_normalized_mutual_information_is_finite():
    first = torch.softmax(torch.randn(4, 16), dim=1)
    second = torch.softmax(torch.randn(4, 16), dim=1)
    assert torch.isfinite(mutual_info(first, second))


def test_generation_pipeline_writes_images_and_manifest(tmp_path):
    input_dir = tmp_path / "input"
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    checkpoint_dir.mkdir()
    Image.new("RGB", (80, 60), color=(120, 140, 160)).save(input_dir / "road.png")

    args = SimpleNamespace(
        mode="syn",
        device="cpu",
        img_size=64,
        style_dim=16,
        latent_dim=8,
        num_domains=2,
        w_hpf=0,
        checkpoint_dir=str(checkpoint_dir),
        resume_iter=1,
        out_dir=str(output_dir),
        num_outs_per_domain=2,
        target_domain=1,
        seed=777,
        lambda_ds=0.25,
        ds_iter=10,
    )
    _, ema = build_model(args)
    torch.save(
        {name: module.state_dict() for name, module in ema.items()},
        checkpoint_dir / "000001_nets_ema.ckpt",
    )

    loader = get_test_loader(input_dir, img_size=64, batch_size=1, num_workers=0, seed=777)
    solver = Solver(args)
    solver._save_training_state(1, Munch(src=loader, ref=loader))
    args.lambda_ds = 0.0
    solver._load_training_state(1, Munch(src=loader, ref=loader))
    assert args.lambda_ds == 0.25
    solver.synthesis(Munch(val=loader))

    assert (output_dir / "road" / "clear.png").is_file()
    assert (output_dir / "road" / "rain_00.png").is_file()
    assert (output_dir / "road" / "rain_01.png").is_file()
    assert len((output_dir / "manifest.csv").read_text().splitlines()) == 3
