"""CPU contract tests; run on the server, not the Windows source workspace."""

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from wm_batch import infer_joint_batch, install_batch_inference, _noise


class ToyVAE:
    temporal_downsample_factor = 4
    upsampling_factor = 16
    z_dim = 3

    def __init__(self):
        self.model = self
        self.encode_batches = []
        self.decode_batches = []
        self.clears = 0

    def clear_cache(self):
        self.clears += 1

    def single_encode(self, video, device):
        self.encode_batches.append(video.shape[0])
        return video[:, :, ::4].mean(dim=(-1, -2), keepdim=True)

    def single_decode(self, latent, device):
        self.decode_batches.append(latent.shape[0])
        pixels = torch.cat([latent[:, :, :1], latent[:, :, 1:].repeat_interleave(4, dim=2)], dim=2)
        return pixels.expand(-1, -1, -1, 16, 16).clamp(-1, 1)


class ToyScheduler:
    def build_inference_schedule(self, *, num_inference_steps, device, dtype, shift_override):
        return torch.linspace(1, 0, num_inference_steps), torch.full((num_inference_steps,), 0.1)

    def step(self, prediction, delta, latent):
        return latent - delta * prediction


class ToyModel:
    device = "cpu"
    torch_dtype = torch.float32
    proprio_dim = 14
    num_hist_frames = 2

    def __init__(self):
        self.vae = ToyVAE()
        self.action_expert = SimpleNamespace(action_dim=14)
        self.video_expert = SimpleNamespace(fuse_vae_embedding_in_latents=False)
        self.infer_video_scheduler = ToyScheduler()
        self.infer_action_scheduler = ToyScheduler()
        self.calls = []

    def eval(self):
        return self

    def infer_joint(self, **kwargs):
        raise AssertionError("The batched path must not loop over upstream B1")

    def _check_resize_height_width(self, height, width, frames):
        return height, width, frames

    def _normalize_infer_action_condition(self, action, *, batch_size, action_horizon, name):
        if tuple(action.shape) != (batch_size, action_horizon, 14):
            raise ValueError("action shape")
        return action

    def _append_proprio_to_context(self, *, context, context_mask, proprio):
        return (torch.cat([context, proprio.unsqueeze(1)], dim=1),
                torch.cat([context_mask, torch.ones((len(proprio), 1), dtype=torch.bool)], dim=1))

    def encode_prompt(self, prompts):
        batch = 1 if isinstance(prompts, str) else len(prompts)
        return torch.zeros(batch, 2, 14), torch.ones(batch, 2, dtype=torch.bool)

    def _predict_joint_noise(self, **kwargs):
        self.calls.append({
            "batch": kwargs["latents_video"].shape[0],
            "context": kwargs["context"].clone(),
            "condition": kwargs["latents_video"][:, :, :2].clone(),
            "timestep_batch": kwargs["timestep_video"].shape[0],
            "action": kwargs["gt_action"].clone(),
        })
        context_signal = kwargs["context"].sum(dim=(1, 2)) * 0.001
        return (kwargs["latents_video"] * 0.03 + context_signal[:, None, None, None, None],
                kwargs["latents_action"] * 0.07 + context_signal[:, None, None])


def inputs(batch, seeds=None):
    return dict(
        input_image=torch.arange(batch, dtype=torch.float32)[:, None, None, None].expand(-1, 3, 16, 16) / 32,
        seeds=list(range(100, 100 + batch)) if seeds is None else seeds,
        action=torch.arange(batch, dtype=torch.float32)[:, None, None].expand(-1, 32, 14) / 16,
        proprio=torch.arange(batch, dtype=torch.float32)[:, None].expand(-1, 14) / 8,
        context=torch.zeros(1, 2, 14), context_mask=torch.ones(1, 2, dtype=torch.bool),
    )


class BatchInferenceTests(unittest.TestCase):
    def test_16_samples_are_ten_true_shared_model_batches(self):
        model = ToyModel()
        result = infer_joint_batch(model, **inputs(16))
        self.assertEqual([call["batch"] for call in model.calls], [16] * 10)
        self.assertEqual([call["timestep_batch"] for call in model.calls], [16] * 10)
        self.assertEqual(model.vae.encode_batches, [16, 16])
        self.assertEqual(model.vae.decode_batches, [16])
        self.assertEqual(result["action"].shape, (16, 32, 14))
        self.assertEqual([len(video) for video in result["video"]], [9] * 16)
        self.assertEqual(result["denoiser_batch_sizes"], [16] * 10)
        self.assertGreater(model.vae.clears, 0)

    def test_seed_streams_match_B1_and_ignore_batch_grouping(self):
        model = ToyModel()
        seeds = [37, 9, 44]
        batched = _noise(model, (3, 4, 1, 1), seeds, "cpu")
        for row, seed in enumerate(seeds):
            expected = torch.randn((1, 3, 4, 1, 1), generator=torch.Generator().manual_seed(seed))
            self.assertTrue(torch.equal(batched[row:row + 1], expected))
        previous_rng = torch.random.get_rng_state().clone()
        result = infer_joint_batch(model, **inputs(3, seeds))
        self.assertTrue(torch.equal(previous_rng, torch.random.get_rng_state()))
        reordered_inputs = inputs(3, seeds)
        order = [2, 0, 1]
        for key in ("input_image", "action", "proprio"):
            reordered_inputs[key] = reordered_inputs[key][order]
        reordered_inputs["seeds"] = [seeds[index] for index in order]
        reordered = infer_joint_batch(ToyModel(), **reordered_inputs)
        for new_row, old_row in enumerate(order):
            self.assertTrue(torch.equal(reordered["action"][new_row], result["action"][old_row]))
            np.testing.assert_array_equal(np.asarray(reordered["video"][new_row][-1]),
                                          np.asarray(result["video"][old_row][-1]))
        # One row of a short tail must keep its own prompt/proprio/seed.
        single = inputs(3, seeds)
        for key in ("input_image", "action", "proprio"):
            single[key] = single[key][1:2]
        single["seeds"] = [seeds[1]]
        tail = infer_joint_batch(ToyModel(), **single)
        self.assertTrue(torch.equal(tail["action"][0], result["action"][1]))
        np.testing.assert_array_equal(np.asarray(tail["video"][0][-1]), np.asarray(result["video"][1][-1]))

    def test_private_proprio_and_conditions_are_never_mixed(self):
        model = ToyModel()
        data = inputs(3)
        infer_joint_batch(model, **data)
        first = model.calls[0]
        self.assertTrue(torch.equal(first["context"][:, -1], data["proprio"]))
        self.assertTrue(torch.equal(first["action"], data["action"]))
        expected = data["input_image"].mean(dim=(-1, -2), keepdim=True).unsqueeze(2).expand(-1, -1, 2, -1, -1)
        for call in model.calls:
            self.assertTrue(torch.equal(call["condition"], expected))

    def test_explicit_history_has_separate_current_frame(self):
        model = ToyModel()
        data = inputs(2)
        data["input_images"] = [torch.full_like(data["input_image"], 0.25)] * 4
        infer_joint_batch(model, **data)
        clean = model.calls[0]["condition"]
        self.assertTrue(torch.equal(clean[:, :, 0], torch.full_like(clean[:, :, 0], 0.25)))
        self.assertTrue(torch.equal(clean[:, :, 1], data["input_image"].mean(dim=(-1, -2), keepdim=True)))

    def test_invalid_contracts_reject_before_denoising(self):
        for update in ({"seeds": [1]}, {"seeds": [True, 2]}, {"num_inference_steps": 1}, {"tiled": True}):
            model = ToyModel()
            data = inputs(2)
            data.update(update)
            with self.assertRaises(ValueError):
                infer_joint_batch(model, **data)
            self.assertEqual(model.calls, [])
        with self.assertRaises(ValueError):
            infer_joint_batch(ToyModel(), **inputs(17))

    def test_install_preserves_original_B1_entry(self):
        model = ToyModel()
        original = model.infer_joint
        install_batch_inference(model)
        self.assertEqual(model.infer_joint, original)
        self.assertEqual(model.infer_joint_B1, original)
        output = model.infer_joint_batch(**inputs(1))
        self.assertEqual(output["batch_size"], 1)


if __name__ == "__main__":
    unittest.main()
