import csv
import datetime
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from munch import Munch
from tqdm import tqdm

import core.utils as utils
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
from core.model import build_model


class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device(args.device)
        self.nets, self.nets_ema = build_model(args)

        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + "_ema", module)

        if args.mode == "train":
            self.optims = Munch()
            for net in self.nets.keys():
                if net == "fan":
                    continue
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == "mapping_network" else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay,
                )

            self.ckptios = [
                CheckpointIO(
                    str(Path(args.checkpoint_dir) / "{:06d}_nets.ckpt"),
                    device=self.device,
                    **self.nets,
                ),
                CheckpointIO(
                    str(Path(args.checkpoint_dir) / "{:06d}_nets_ema.ckpt"),
                    device=self.device,
                    **self.nets_ema,
                ),
                CheckpointIO(
                    str(Path(args.checkpoint_dir) / "{:06d}_optims.ckpt"),
                    device=self.device,
                    **self.optims,
                ),
            ]
        else:
            self.ckptios = [
                CheckpointIO(
                    str(Path(args.checkpoint_dir) / "{:06d}_nets_ema.ckpt"),
                    device=self.device,
                    **self.nets_ema,
                )
            ]

        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ("ema" not in name) and ("fan" not in name):
                print(f"Initializing {name}...")
                network.apply(utils.he_init)

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _training_state_path(self, step):
        return Path(self.args.checkpoint_dir) / f"{step:06d}_training_state.ckpt"

    def _save_training_state(self, step, loaders):
        numpy_state = np.random.get_state()
        loader_generators = {}
        for name, loader in (("source", loaders.src), ("reference", loaders.ref)):
            loader_generators[name] = {}
            if loader.generator is not None:
                loader_generators[name]["loader"] = loader.generator.get_state()
            sampler_generator = getattr(loader.sampler, "generator", None)
            if sampler_generator is not None:
                loader_generators[name]["sampler"] = sampler_generator.get_state()
        state = {
            "python_random": random.getstate(),
            "numpy": {
                "algorithm": numpy_state[0],
                "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
                "position": numpy_state[2],
                "has_gauss": numpy_state[3],
                "cached_gaussian": numpy_state[4],
            },
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "lambda_ds": self.args.lambda_ds,
            "loader_generators": loader_generators,
        }
        destination = self._training_state_path(step)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(state, temporary)
        os.replace(temporary, destination)

    def _load_training_state(self, step, loaders):
        source = self._training_state_path(step)
        if not source.is_file():
            initial = self.args.lambda_ds
            self.args.lambda_ds = max(0.0, initial * (1.0 - step / self.args.ds_iter))
            print(
                f"Warning: {source} is absent; RNG state cannot be restored from this "
                "legacy checkpoint."
            )
            return

        state = torch.load(source, map_location="cpu", weights_only=True)
        random.setstate(state["python_random"])
        numpy_state = state["numpy"]
        np.random.set_state(
            (
                numpy_state["algorithm"],
                numpy_state["keys"].cpu().numpy().astype(np.uint32),
                numpy_state["position"],
                numpy_state["has_gauss"],
                numpy_state["cached_gaussian"],
            )
        )
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state["cuda"]:
            torch.cuda.set_rng_state_all(state["cuda"])
        self.args.lambda_ds = float(state["lambda_ds"])
        for name, loader in (("source", loaders.src), ("reference", loaders.ref)):
            generator_states = state["loader_generators"].get(name, {})
            if "loader" in generator_states and loader.generator is not None:
                loader.generator.set_state(generator_states["loader"])
            sampler_generator = getattr(loader.sampler, "generator", None)
            if "sampler" in generator_states and sampler_generator is not None:
                sampler_generator.set_state(generator_states["sampler"])

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad(set_to_none=True)

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims
        initial_lambda_ds = args.lambda_ds

        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)
            self._load_training_state(args.resume_iter, loaders)

        fetcher = InputFetcher(
            loaders.src, loaders.ref, args.latent_dim, "train", device=self.device
        )
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, "val", device=self.device)
        inputs_val = next(fetcher_val)

        print("Start training...")
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            # fetch images and labels

            inputs = next(fetcher)

            labels = torch.full(
                (inputs.x_src.size(0),),
                args.target_domain,
                dtype=torch.long,
                device=self.device,
            )

            x_gt, x_rain = (
                inputs.x_src[:, :, :, : args.img_size],
                inputs.x_src[:, :, :, args.img_size :],
            )
            y_gt, y_rain = (
                inputs.x_ref[:, :, :, : args.img_size],
                inputs.x_ref[:, :, :, args.img_size :],
            )

            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2

            masks = nets.fan.get_heatmap(x_gt) if args.w_hpf > 0 else None

            d_loss, d_losses_latent = compute_d_loss(
                nets, args, x_gt, x_rain, labels, z_trg=z_trg, masks=masks
            )
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            d_loss, d_losses_ref = compute_d_loss(nets, args, x_gt, x_rain, labels, masks=masks)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            g_loss, g_losses_latent = compute_g_loss(
                nets, args, x_gt, x_rain, y_gt, y_rain, labels, z_trgs=[z_trg, z_trg2], masks=masks
            )
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.mapping_network.step()
            optims.style_encoder.step()

            g_loss, g_losses_ref = compute_g_loss(
                nets, args, x_gt, x_rain, y_gt, y_rain, labels, masks=masks
            )
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()

            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
            moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

            if args.lambda_ds > 0:
                args.lambda_ds = max(0.0, args.lambda_ds - (initial_lambda_ds / args.ds_iter))

            if (i + 1) % args.save_every == 0:
                self._save_checkpoint(step=i + 1)
                self._save_training_state(step=i + 1, loaders=loaders)

            if (i + 1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i + 1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip(
                    [d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                    ["D/latent_", "D/ref_", "G/latent_", "G/ref_"],
                    strict=False,
                ):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses["G/lambda_ds"] = args.lambda_ds
                log += " ".join([f"{key}: [{value:.4f}]" for key, value in all_losses.items()])
                print(log)

            if (i + 1) % args.sample_every == 0:
                Path(args.sample_dir).mkdir(parents=True, exist_ok=True)
                utils.debug_image(nets_ema, args, inputs=inputs_val, step=i + 1)

    @torch.no_grad()
    def sample(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        self._load_checkpoint(args.resume_iter)

        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, "val", device=self.device)
        inputs_val = next(fetcher_val)
        Path(args.sample_dir).mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(min(2, len(loaders.val)))):
            utils.debug_image(nets_ema, args, inputs=inputs_val, step=i)
            inputs_val = next(fetcher_val)

    @torch.no_grad()
    def synthesis(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        self._load_checkpoint(args.resume_iter)
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, "val", device=self.device)
        output_dir = Path(args.out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for _ in tqdm(range(len(loaders.val)), desc="Generating rain"):
            records.extend(utils.synthesis_images(nets_ema, args, next(fetcher_val)))

        manifest = output_dir / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = ["source", "output", "variant", "seed", "checkpoint_iteration"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"Generated {len(records)} images. Manifest: {manifest}")


def compute_d_loss(nets, args, x_gt, x_rain, labels, z_trg=None, masks=None):
    x_rain.requires_grad_()
    out = nets.discriminator(x_rain, labels)
    loss_reg = r1_reg(out, x_rain)

    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, labels)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_rain, labels)

        x_fake = nets.generator(x_gt, s_trg, masks=masks)
    out2 = nets.discriminator(x_fake, labels)

    # Rain streak-guided loss
    if args.rain_streak_guidance:
        weight = args.rain_streak_guidance_weight
        loss_real = adv_loss(out - (weight * out2), 1)
        loss_fake = adv_loss(out2 - (weight * out), 0)
    else:
        loss_real = adv_loss(out, 1)
        loss_fake = adv_loss(out2, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(), fake=loss_fake.item(), reg=loss_reg.item())


def compute_g_loss(nets, args, x_gt, x_rain, y_gt, y_rain, labels, z_trgs=None, masks=None):
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs

    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, labels)
    else:
        s_trg = nets.style_encoder(x_rain, labels)

    x_fake = nets.generator(x_gt, s_trg, masks=masks)

    out = nets.discriminator(x_fake, labels)

    loss_adv = adv_loss(out, 1)

    s_pred = nets.style_encoder(x_fake, labels)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, labels)
        s_trg4 = nets.style_encoder(y_rain, labels)
        loss_map = torch.mean(torch.abs(s_trg2 - s_trg4))
    else:
        s_trg2 = nets.style_encoder(y_rain, labels)
        loss_map = torch.zeros((), device=x_gt.device)

    x_fake2 = nets.generator(x_gt, s_trg2, masks=masks)
    x_fake2 = x_fake2.detach()

    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    loss_npmi = mutual_info(F.softmax(s_trg, dim=1), F.softmax(s_trg2, dim=1))
    zero_style = torch.zeros_like(s_trg2)

    x_ori = nets.generator(x_gt, zero_style, masks=masks)
    loss_bias = torch.mean(torch.abs(x_ori - x_gt))

    loss = (
        loss_adv
        + args.lambda_sty * loss_sty
        + loss_map
        + args.lambda_npmi * loss_npmi
        + args.gt * loss_bias
        - args.lambda_ds * loss_ds
    )

    return loss, Munch(
        adv=loss_adv.item(),
        sty=loss_sty.item(),
        ds=loss_ds.item(),
        mapp=loss_map.item(),
        bias=loss_bias.item(),
        MI=loss_npmi.item(),
    )


def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)

    bn, k = x_out.size()
    assert x_tf_out.size(0) == bn and x_tf_out.size(1) == k

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.0  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def mutual_info(x_out, x_tf_out, EPS=1e-10):
    _, k = x_out.size()

    p_i_j = compute_joint(x_out, x_tf_out)
    assert p_i_j.size() == (k, k)

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k).clone()
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k).clone()  # but should be same, symmetric

    p_i_j = p_i_j.clamp_min(EPS)
    p_j = p_j.clamp_min(EPS)
    p_i = p_i.clamp_min(EPS)

    loss = torch.log(p_i_j / (p_j * p_i)) / (-torch.log(p_i_j))

    loss = loss.mean()

    return loss


@torch.no_grad()
def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters(), strict=False):
        param_test.lerp_(param, 1 - beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert grad_dout2.size() == x_in.size()
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg
