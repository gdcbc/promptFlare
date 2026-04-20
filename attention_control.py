import torch
import torch.nn.functional as F


class CrossAttentionRecorder:
    def __init__(self):
        self.mode = "full"   # "full" or "bos"
        self.full_features = {}
        self.bos_features = {}

    def clear(self):
        self.full_features.clear()
        self.bos_features.clear()

    def set_mode(self, mode: str):
        assert mode in ["full", "bos"]
        self.mode = mode

    def add(self, name: str, feat: torch.Tensor):
        if self.mode == "full":
            self.full_features[name] = feat
        else:
            self.bos_features[name] = feat

    def mean_loss(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        losses = []
        for name in self.full_features:
            if name in self.bos_features:
                losses.append(
                    F.mse_loss(
                        self.full_features[name].float(),
                        self.bos_features[name].float()
                    )
                )
        if len(losses) == 0:
            return torch.zeros((), device=device, dtype=dtype)
        return torch.stack(losses).mean()


class FlareI2IAttnProcessor:
    def __init__(self, name: str, recorder: CrossAttentionRecorder, enabled: bool = True):
        self.name = name
        self.recorder = recorder
        self.enabled = enabled

    @staticmethod
    def _reshape_in(hidden_states: torch.Tensor):
        input_ndim = hidden_states.ndim
        height = width = None
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
            height, width = h, w
        return hidden_states, input_ndim, height, width

    @staticmethod
    def _reshape_out(hidden_states: torch.Tensor, input_ndim: int, height: int, width: int):
        if input_ndim == 4:
            b, n, c = hidden_states.shape
            hidden_states = hidden_states.transpose(1, 2).reshape(b, c, height, width)
        return hidden_states

    @staticmethod
    def _attention_core(attn, query, key, value, bos_only: bool) -> torch.Tensor:
        scale = attn.scale if hasattr(attn, "scale") else (query.shape[-1] ** -0.5)
        scores = torch.bmm(query, key.transpose(1, 2)) * scale

        if bos_only and scores.shape[-1] > 1:
            scores[:, :, 1:] = -1e4

        probs = torch.softmax(scores, dim=-1)
        hidden_states = torch.bmm(probs, value)
        return hidden_states

    def _run_attention(self, attn, hidden_states, encoder_hidden_states, bos_only: bool):
        residual = hidden_states
        hidden_states, input_ndim, height, width = self._reshape_in(hidden_states)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        hidden_states = self._attention_core(attn, query, key, value, bos_only=bos_only)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states = self._reshape_out(hidden_states, input_ndim, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        if encoder_hidden_states is None or not self.enabled:
            return self._run_attention(attn, hidden_states, encoder_hidden_states, bos_only=False)

        bos_only = (self.recorder.mode == "bos")
        out = self._run_attention(attn, hidden_states, encoder_hidden_states, bos_only=bos_only)

        # 记录当前层输出，用于 full / bos 的差异损失
        self.recorder.add(self.name, out)
        return out