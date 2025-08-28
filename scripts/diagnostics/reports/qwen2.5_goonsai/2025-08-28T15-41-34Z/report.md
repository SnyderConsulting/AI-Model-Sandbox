# Qwen2.5-3B-goonsai-nsfw-100k Architecture Report
Model id: goonsai-com/civitaiprompts/qwen2.5-3B-goonsai-nsfw-100k

## Config
```json
{
  "vocab_size": 151936,
  "max_position_embeddings": 32768,
  "hidden_size": 2048,
  "intermediate_size": 11008,
  "num_hidden_layers": 36,
  "num_attention_heads": 16,
  "use_sliding_window": false,
  "sliding_window": null,
  "max_window_layers": 70,
  "num_key_value_heads": 2,
  "hidden_act": "silu",
  "initializer_range": 0.02,
  "rms_norm_eps": 1e-06,
  "use_cache": false,
  "rope_theta": 1000000.0,
  "rope_scaling": null,
  "attention_dropout": 0.0,
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention"
  ],
  "return_dict": true,
  "output_hidden_states": false,
  "torchscript": false,
  "torch_dtype": "bfloat16",
  "pruned_heads": {},
  "tie_word_embeddings": true,
  "chunk_size_feed_forward": 0,
  "is_encoder_decoder": false,
  "is_decoder": false,
  "cross_attention_hidden_size": null,
  "add_cross_attention": false,
  "tie_encoder_decoder": false,
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "finetuning_task": null,
  "id2label": {
    "0": "LABEL_0",
    "1": "LABEL_1"
  },
  "label2id": {
    "LABEL_0": 0,
    "LABEL_1": 1
  },
  "task_specific_params": null,
  "problem_type": null,
  "tokenizer_class": null,
  "prefix": null,
  "bos_token_id": 151643,
  "pad_token_id": null,
  "eos_token_id": 151645,
  "sep_token_id": null,
  "decoder_start_token_id": null,
  "max_length": 20,
  "min_length": 0,
  "do_sample": false,
  "early_stopping": false,
  "num_beams": 1,
  "num_beam_groups": 1,
  "diversity_penalty": 0.0,
  "temperature": 1.0,
  "top_k": 50,
  "top_p": 1.0,
  "typical_p": 1.0,
  "repetition_penalty": 1.0,
  "length_penalty": 1.0,
  "no_repeat_ngram_size": 0,
  "encoder_no_repeat_ngram_size": 0,
  "bad_words_ids": null,
  "num_return_sequences": 1,
  "output_scores": false,
  "return_dict_in_generate": false,
  "forced_bos_token_id": null,
  "forced_eos_token_id": null,
  "remove_invalid_values": false,
  "exponential_decay_length_penalty": null,
  "suppress_tokens": null,
  "begin_suppress_tokens": null,
  "_name_or_path": "goonsai-com/civitaiprompts",
  "transformers_version": "4.55.2",
  "model_type": "qwen2",
  "tf_legacy_loss": false,
  "use_bfloat16": false,
  "output_attentions": false
}
```

## Tokenizer
vocab_size=151936, bos=None, eos=151645, pad=151645, tokenizer_class=Qwen2TokenizerFast

## Model core
hidden_size=2048, num_hidden_layers=36, num_attention_heads=16, num_key_value_heads=2, head_dim=128, intermediate_size=11008, activation=silu, rms_norm_eps=1e-06
rope_theta=1000000.0, rope_scaling=None

## Parameters
total_params: 3,397,103,616
~memory (fp16): 6.33 GB, ~memory (bf16): 6.33 GB
default/declared dtype: bfloat16

### By kind
- mlp.gate_proj: 811,597,824
- mlp.up_proj: 811,597,824
- mlp.down_proj: 811,597,824
- embeddings: 311,164,928
- lm_head: 311,164,928
- attn.q_proj: 151,068,672
- attn.o_proj: 150,994,944
- attn.k_proj: 18,883,584
- attn.v_proj: 18,883,584
- pre_attn_norm: 73,728
- post_attn_norm: 73,728
- other: 2,048

### Per layer param counts
- layer 00: 77,076,992
- layer 01: 77,076,992
- layer 02: 77,076,992
- layer 03: 77,076,992
- layer 04: 77,076,992
- layer 05: 77,076,992
- layer 06: 77,076,992
- layer 07: 77,076,992
- layer 08: 77,076,992
- layer 09: 77,076,992
- layer 10: 77,076,992
- layer 11: 77,076,992
- layer 12: 77,076,992
- layer 13: 77,076,992
- layer 14: 77,076,992
- layer 15: 77,076,992
- layer 16: 77,076,992
- layer 17: 77,076,992
- layer 18: 77,076,992
- layer 19: 77,076,992
- layer 20: 77,076,992
- layer 21: 77,076,992
- layer 22: 77,076,992
- layer 23: 77,076,992
- layer 24: 77,076,992
- layer 25: 77,076,992
- layer 26: 77,076,992
- layer 27: 77,076,992
- layer 28: 77,076,992
- layer 29: 77,076,992
- layer 30: 77,076,992
- layer 31: 77,076,992
- layer 32: 77,076,992
- layer 33: 77,076,992
- layer 34: 77,076,992
- layer 35: 77,076,992
