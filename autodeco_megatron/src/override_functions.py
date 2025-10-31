from typing import OrderedDict

from megatron.core import parallel_state as mcore_ps
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk


def _postprocess(
        self,
        hidden_states,
        input_ids,
        position_ids,
        labels,
        rotary_pos_emb,
        rotary_pos_cos,
        rotary_pos_sin,
        mtp_in_postprocess=None,
        loss_mask=None,
        decoder_input=None,
        attention_mask=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        runtime_gather_output=None,
        extra_block_kwargs=None,
        inference_context=None,
):
    """Postprocesses decoder hidden states to generate logits or compute loss.

    Applies Multi-Token Prediction if enabled, generates output logits through
    the output layer, and computes language model loss when labels are provided.
    """

    # logits and loss
    output_weight = None
    if self.share_embeddings_and_output_weights:
        output_weight = self.shared_embedding_or_output_weight()

    is_last_stage = mcore_ps.is_pipeline_last_stage()

    if not is_last_stage:
        return hidden_states

    logits, _ = self.output_layer(
        hidden_states, weight=output_weight, runtime_gather_output=True
    )

    if has_config_logger_enabled(self.config):
        payload = OrderedDict(
            {
                'input_ids': input_ids,
                'position_ids': position_ids,
                'attention_mask': attention_mask,
                'decoder_input': decoder_input,
                'logits': logits,
            }
        )
        log_config_to_disk(self.config, payload, prefix='input_and_logits')

    hidden_states = hidden_states.transpose(0, 1).contiguous()
    logits = logits.transpose(0, 1).contiguous()
    temp_logits, top_p_logits = self.adapter(logits=logits, hidden_states=hidden_states)
    return {
        "logits": logits,
        "hidden_states": hidden_states,
        "temp_logits": temp_logits,
        "top_p_logits": top_p_logits
    }
