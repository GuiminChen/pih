import json
from pathlib import Path
import runpy
import unittest

ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = runpy.run_path(str(ROOT / "python/pih_client/preflight.py"))
MODEL = runpy.run_path(str(ROOT / "tools/inspect_deepseek_v41_config.py"))


class B300PreflightTests(unittest.TestCase):
    def receipt(self, count=8, name="NVIDIA B300", capability="10.3", memory=288000,
                profile="b300"):
        rows = "\n".join(
            f"{i}, GPU-11111111-1111-1111-1111-{i:012x}, {name}, {capability}, {memory}"
            for i in range(count)
        )
        return PREFLIGHT["compile_target_host_preflight"](
            hardware_profile=profile, devices=tuple(range(count)),
            nvidia_smi_output=rows, meminfo="MemTotal: 1073741824 kB\n",
            system="Linux", environment={},
        )

    def test_single_and_eight_gpu_receipts_replay(self):
        for count in (1, 8):
            receipt = self.receipt(count)
            self.assertEqual(receipt.payload["world_size"], count)
            self.assertEqual(receipt.payload["support_state"], "hardware_evidence_open")
            PREFLIGHT["verify_target_host_preflight"](receipt)
            document = json.loads(json.dumps({**receipt.to_payload(),
                                              "receipt_root": receipt.receipt_root}))
            self.assertEqual(PREFLIGHT["decode_target_host_preflight_document"](document), receipt)

    def test_other_blackwell_devices_are_not_b300(self):
        for name in ("NVIDIA B200", "NVIDIA GB300", "NVIDIA B300 MIG 1g"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.receipt(name=name)

    def test_wrong_architecture_or_memory_rejected(self):
        for args in ({"capability": "10.0"}, {"capability": "9.0"},
                     {"memory": 180000}, {"memory": 600000}, {"count": 9}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.receipt(**args)

    def test_h100_limit_stays_four(self):
        with self.assertRaisesRegex(ValueError, "1-4"):
            self.receipt(count=8, profile="h100-pcie", name="NVIDIA H100 PCIe",
                         capability="9.0", memory=81559)


class V41ConfigTests(unittest.TestCase):
    def config(self):
        # Minimal diagnostic fixture; not a runnable checkpoint/configuration.
        return {"model_type": "deepseek_v41", "architectures": ["DeepseekV41ForCausalLM"],
                "text_config": {"model_type": "deepseek_v41_text", "num_hidden_layers": 40,
                                "hidden_size": 5120, "vocab_size": 129280,
                                "n_routed_experts": 384, "num_experts_per_tok": 6,
                                "max_position_embeddings": 1048576}, "vision_config": {}}

    def inspect(self, config):
        return MODEL["inspect_config"](json.dumps(config).encode())

    def test_recognition_does_not_authorize_execution(self):
        result = self.inspect(self.config())
        self.assertEqual(result["model_id"], "deepseek-ai/DeepSeek-V4.1-Flash")
        self.assertEqual(result["support_status"], "unsupported")
        self.assertEqual(result["native_runtime_status"], "source_implemented_unqualified")
        self.assertFalse(result["source_revision_verified"])
        self.assertEqual(len(result["config_sha256"]), 64)

    def test_legacy_model_and_shapes_rejected(self):
        config = self.config()
        config["model_type"] = "deepseek_v4"
        with self.assertRaises(ValueError):
            self.inspect(config)
        for value in (43, True, 40.0):
            config = self.config()
            config["text_config"]["num_hidden_layers"] = value
            with self.assertRaises(ValueError):
                self.inspect(config)

    def test_strict_bounded_json(self):
        for payload in (b'', b'[]', b'{"x":1,"x":2}', b'{"x":NaN}', b'\xff',
                        b' ' * (MODEL["MAX_CONFIG_BYTES"] + 1)):
            with self.subTest(payload=payload[:30]), self.assertRaises(ValueError):
                MODEL["inspect_config"](payload)


if __name__ == "__main__":
    unittest.main()
