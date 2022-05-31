# Codes for ICLR-22 work
# 'Differentiable Gradient Sampling for Learning Implicit 3D Scene Reconstructions from a Single Image'
# Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com

import torch
import numpy as np
import torch.autograd.profiler as profiler


class PositionalEncoding(torch.nn.Module):

    # The implementation of this method if borrowed from PixelNeRF
    # https://github.com/sxyu/pixel-nerf/blob/master/src/model/code.py#L11
    def __init__(self, num_freqs=6, d_in=3, freq_factor=np.pi, include_input=True,
                 **kwargs):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

    def forward(self, x, forwardMode):
        if forwardMode in ['valOnly']:
            return self.forwardValOnly(x)
        elif forwardMode in ['gradTrack']:
            return self.forwardReturnGradTrack(x)
        else:
            raise NotImplementedError('Unknown forwardMode: %s' % self.forwardMode)

    # The implementation of this method is borrowed from PixelNeRF
    # https://github.com/sxyu/pixel-nerf/blob/master/src/model/code.py#L30
    def forwardValOnly(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (batch, self.d_in)
        :return (batch, self.d_out)
        """
        with profiler.record_function("positional_enc"):
            embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
            embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
            embed = embed.view(x.shape[0], -1)

            if self.include_input:
                embed = torch.cat((x, embed), dim=-1)
            return embed

    def forwardReturnGradTrack(self, x):
        # Input:
        #   x: (b, d_in(3))  # The same as above
        # Output:
        #   outFour: (b, d_out(39), 4(val1 + grad3))

        embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
        val = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
        gradDiag = self._freqs * torch.cos(torch.addcmul(
            self._phases, embed, self._freqs))
        gradThree = torch.cat([
            gradDiag[..., 0:1], torch.zeros_like(gradDiag),
            gradDiag[..., 1:2], torch.zeros_like(gradDiag),
            gradDiag[..., 2:3],
        ], -1).contiguous().view(gradDiag.shape[0], gradDiag.shape[1], 3, 3)

        val = val.view(x.shape[0], -1)
        gradThree = gradThree.view(x.shape[0], self.num_freqs * 2 * 3, 3)

        if self.include_input:
            val = torch.torch.cat((x, val), dim=1)
            tmp = torch.eye(3, dtype=gradThree.dtype, device=gradThree.device).repeat(x.shape[0], 1, 1)
            gradThree = torch.cat([tmp, gradThree], dim=1)

        outFour = torch.cat([val[:, :, None], gradThree], dim=2)
        return outFour
