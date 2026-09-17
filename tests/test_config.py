from core.config import load_config, solver_args


def test_paper_config_maps_to_solver_arguments():
    config = load_config("configs/paper.yaml")
    args = solver_args(config, mode="train")

    assert args.img_size == 256
    assert args.latent_dim == 16
    assert args.style_dim == 64
    assert args.total_iters == 100000
    assert args.seed == 777
    assert args.rain_streak_guidance is True


def test_cli_overrides_do_not_mutate_config():
    config = load_config("configs/paper.yaml")
    args = solver_args(config, mode="train", batch_size=2, total_iterations=3)

    assert args.batch_size == 2
    assert args.total_iters == 3
    assert config["training"]["batch_size"] == 8
