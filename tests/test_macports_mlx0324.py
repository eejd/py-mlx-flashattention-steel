import mlx.core as mx
import numpy as np
import pytest

from mlx_mfa import flash_attention, _ext


@pytest.mark.parametrize('dtype', [mx.float16, mx.bfloat16])
@pytest.mark.parametrize('causal', [False, True])
def test_attention_non_aligned(dtype, causal):
    mx.random.seed(17)
    q = mx.random.normal((1, 2, 37, 64)).astype(dtype)
    k = mx.random.normal((1, 2, 37, 64)).astype(dtype)
    v = mx.random.normal((1, 2, 37, 64)).astype(dtype)
    actual = flash_attention(q, k, v, causal=causal, backend='mfa')
    scores = (q.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2)) * (64 ** -0.5)
    if causal:
        mask = mx.arange(37)[:, None] >= mx.arange(37)[None, :]
        scores = mx.where(mask, scores, -1e9)
    expected = mx.softmax(scores, axis=-1) @ v.astype(mx.float32)
    mx.eval(actual, expected)
    np.testing.assert_allclose(np.array(actual.astype(mx.float32)), np.array(expected), atol=0.04, rtol=0.04)


def test_smooth_quantize_mean_buffer():
    x = mx.arange(1 * 2 * 256 * 128).reshape(1, 2, 256, 128).astype(mx.float32)
    x = (mx.sin(x * 0.01) * 2).astype(mx.float16)
    quantized, scales, mean = _ext.mfa_smooth_quantize_k(x, 32)
    mx.eval(quantized, scales, mean)
    np.testing.assert_allclose(np.array(mean.astype(mx.float32)), np.array(mx.mean(x.astype(mx.float32), axis=2, keepdims=True)), atol=0.002, rtol=0.02)
