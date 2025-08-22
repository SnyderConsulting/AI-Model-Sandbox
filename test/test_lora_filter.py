from utils.lora import filter_lora_targets


class DummyModel:
    def get_lora_module_names(self):
        return [
            "blocks.0.self_attn.q",
            "blocks.21.cross_attn.k",
            "blocks.21.cross_attn.v",
            "blocks.21.ffn.0",
        ]


def test_filter_excludes_self_attn():
    cfg = {
        "exclude": ["self_attn"],
        "include": ["cross_attn.(k|v)", "ffn.0"],
        "train_blocks_range": [20, 29],
    }
    out = filter_lora_targets(DummyModel(), cfg)
    assert out["target_modules"] == [
        "blocks.21.cross_attn.k",
        "blocks.21.cross_attn.v",
        "blocks.21.ffn.0",
    ]
