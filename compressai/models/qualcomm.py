import torch
import torch.nn as nn

from compressai.layers import PatchEmbed, PatchMerging, PatchSplitting, BasicLayer
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.ops import quantize_ste

from .google import MeanScaleHyperprior
from .utils import conv, deconv

__all__ = [
    "ConvHyperprior",
    "ConvChARM",
    "SwinTHyperprior",
    "SwinTChARM",
]


class ConvHyperprior(MeanScaleHyperprior):
    """Conv-Hyperprior
    Y. Zhu, Y. Yang, T. Cohen:
    "Transformer-Based Transform Coding"

    International Conference on Learning Representations (ICLR), 2022
    https://openreview.net/pdf?id=IDwN6xjHnK8
    """
    def __init__(self, main_dim, hyper_dim, **kwargs):
        super().__init__(hyper_dim, main_dim, **kwargs)

        self.main_dim = main_dim
        self.hyper_dim = hyper_dim

        self.entropy_bottleneck = EntropyBottleneck(self.hyper_dim)

        self.g_a = nn.Sequential(
            conv(3, self.main_dim),
            GDN(self.main_dim),
            conv(self.main_dim, self.main_dim),
            GDN(self.main_dim),
            conv(self.main_dim, self.main_dim),
            GDN(self.main_dim),
            conv(self.main_dim, self.main_dim),
        )

        self.g_s = nn.Sequential(
            deconv(self.main_dim, self.main_dim),
            GDN(self.main_dim, inverse=True),
            deconv(self.main_dim, self.main_dim),
            GDN(self.main_dim, inverse=True),
            deconv(self.main_dim, self.main_dim),
            GDN(self.main_dim, inverse=True),
            deconv(self.main_dim, 3),
        )

        self.h_a = nn.Sequential(
            conv(self.main_dim, self.hyper_dim, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(self.hyper_dim, self.hyper_dim),
            nn.ReLU(inplace=True),
            conv(self.hyper_dim, self.hyper_dim),
        )

        self.h_s = nn.Sequential(
            deconv(self.hyper_dim, self.hyper_dim),
            nn.ReLU(inplace=True),
            deconv(self.hyper_dim, self.hyper_dim),
            nn.ReLU(inplace=True),
            conv(self.hyper_dim, self.main_dim * 2, stride=1, kernel_size=3),
        )

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`.
           Primarily used when a checkpoint is going to be evaluated without prior knowledge of its model configuration.
        """
        main_dim = state_dict["g_a.0.weight"].size(0)
        hyper_dim = state_dict["h_a.0.weight"].size(0)
        net = cls(main_dim, hyper_dim)
        net.load_state_dict(state_dict)
        return net

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = quantize_ste(z - z_offset) + z_offset

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_hat = quantize_ste(y - means_hat) + means_hat
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


class ChARMBlockHalf(nn.Module):
    """Half of ChARM Block as illustrated in Figure 12 (Appendix) of the following paper:
    Y. Zhu, Y. Yang, T. Cohen:
    "Transformer-Based Transform Coding"

    International Conference on Learning Representations (ICLR), 2022
    https://openreview.net/pdf?id=IDwN6xjHnK8
    """
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        c1 = (out_dim - in_dim) // 3 + in_dim
        c2 = 2 * (out_dim - in_dim) // 3 + in_dim
        self.layers = nn.Sequential(
            conv(in_dim, c1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(c1, c2, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(c2, out_dim, kernel_size=3, stride=1)
        )

    def forward(self, x):
        return self.layers(x)


class ConvChARM(ConvHyperprior):
    """Conv-ChARM
    Y. Zhu, Y. Yang, T. Cohen:
    "Transformer-Based Transform Coding"

    International Conference on Learning Representations (ICLR), 2022
    https://openreview.net/pdf?id=IDwN6xjHnK8
    """
    def __init__(self, main_dim, hyper_dim, **kwargs):
        super().__init__(main_dim, hyper_dim, **kwargs)
        self.num_slices = 10

        self.charm_mean_transforms = nn.ModuleList(
            [ChARMBlockHalf(in_dim=32 + 32 * i, out_dim=32) for i in range(self.num_slices)]
        )
        self.charm_scale_transforms = nn.ModuleList(
            [ChARMBlockHalf(in_dim=32 + 32 * i, out_dim=32) for i in range(self.num_slices)]
        )

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = quantize_ste(z - z_offset) + z_offset

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        means_hat_slices = means_hat.chunk(self.num_slices, 1)
        scales_hat_slices = scales_hat.chunk(self.num_slices, 1)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            mean_support = torch.cat([means_hat_slices[slice_index]] + y_hat_slices, dim=1)
            mu = self.charm_mean_transforms[slice_index](mean_support)

            scale_support = torch.cat([scales_hat_slices[slice_index]] + y_hat_slices, dim=1)
            scale = self.charm_scale_transforms[slice_index](scale_support)

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = quantize_ste(y_slice - mu) + mu

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        means_hat_slices = means_hat.chunk(self.num_slices, 1)
        scales_hat_slices = scales_hat.chunk(self.num_slices, 1)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            mean_support = torch.cat([means_hat_slices[slice_index]] + y_hat_slices, dim=1)
            mu = self.charm_mean_transforms[slice_index](mean_support)

            scale_support = torch.cat([scales_hat_slices[slice_index]] + y_hat_slices, dim=1)
            scale = self.charm_scale_transforms[slice_index](scale_support)

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        means_hat_slices = means_hat.chunk(self.num_slices, 1)
        scales_hat_slices = scales_hat.chunk(self.num_slices, 1)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            mean_support = torch.cat([means_hat_slices[slice_index]] + y_hat_slices, dim=1)
            mu = self.charm_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([scales_hat_slices[slice_index]] + y_hat_slices, dim=1)
            scale = self.charm_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}


class SwinTAnalysisTransform(nn.Module):

    def __init__(self, embed_dim, embed_out_dim, depths, head_dim, window_size, input_dim):
        super().__init__()
        self.patch_embed = PatchEmbed(dim=input_dim, out_dim=embed_dim[0])
        # self.patch_embed = nn.Conv2d(input_dim, embed_dim[0], 2, 2)
        num_layers = len(depths)
        self.layers = nn.ModuleList(
            [BasicLayer(dim=embed_dim[i],
                        out_dim=embed_out_dim[i],
                        head_dim=head_dim[i],
                        depth=depths[i],
                        window_size=window_size[i],
                        downsample=PatchMerging if (i < num_layers - 1) else None)
             for i in range(num_layers)]
        )

    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        return x.permute(0, 3, 1, 2)


class SwinTSynthesisTransform(nn.Module):

    def __init__(self, embed_dim, embed_out_dim, depths, head_dim, window_size):
        super().__init__()
        num_layers = len(depths)
        self.layers = nn.ModuleList(
            [BasicLayer(dim=embed_dim[i],
                        out_dim=embed_out_dim[i],
                        head_dim=head_dim[i],
                        depth=depths[i],
                        window_size=window_size[i],
                        downsample=PatchSplitting)
             for i in range(num_layers)]
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        for layer in self.layers:
            x = layer(x)
        return x.permute(0, 3, 1, 2)


class SwinTHyperAnalysisTransform(nn.Module):

    def __init__(self, embed_dim, embed_out_dim, depths, head_dim, window_size, input_dim):
        super().__init__()
        self.patch_merger = PatchEmbed(dim=input_dim, out_dim=embed_out_dim[0])
        # self.patch_merger = nn.Conv2d(input_dim, embed_dim[0], 2, 2)
        num_layers = len(depths)
        self.layers = nn.ModuleList(
            [BasicLayer(dim=embed_dim[i],
                        out_dim=embed_out_dim[i],
                        head_dim=head_dim[i],
                        depth=depths[i],
                        window_size=window_size[i],
                        downsample=PatchMerging if (i < num_layers - 1) else None)
             for i in range(num_layers)]
        )

    def forward(self, x):
        x = self.patch_merger(x)
        for layer in self.layers:
            x = layer(x)
        return x.permute(0, 3, 1, 2)


class SwinTHyperSynthesisTransform(nn.Module):

    def __init__(self, embed_dim, embed_out_dim, depths, head_dim, window_size):
        super().__init__()
        num_layers = len(depths)
        self.layers = nn.ModuleList(
            [BasicLayer(dim=embed_dim[i],
                        out_dim=embed_out_dim[i],
                        head_dim=head_dim[i],
                        depth=depths[i],
                        window_size=window_size[i],
                        downsample=PatchSplitting)
             for i in range(num_layers)]
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        for layer in self.layers:
            x = layer(x)
        return x.permute(0, 3, 1, 2)


class SwinTHyperprior(ConvHyperprior):
    """SwinT-Hyperprior
    Y. Zhu, Y. Yang, T. Cohen:
    "Transformer-Based Transform Coding"

    International Conference on Learning Representations (ICLR), 2022
    https://openreview.net/pdf?id=IDwN6xjHnK8
    """

    def __init__(self, *args, **kwargs):
        super().__init__(main_dim=kwargs['g_a']['embed_dim'][-1], hyper_dim=kwargs['h_a']['embed_dim'][-1])
        self.g_a = SwinTAnalysisTransform(**kwargs['g_a'])
        self.g_s = SwinTSynthesisTransform(**kwargs['g_s'])
        self.h_a = SwinTHyperAnalysisTransform(**kwargs['h_a'])
        self.h_s = SwinTHyperSynthesisTransform(**kwargs['h_s'])

    def _config_from_state_dict(self, state_dict):
        raise NotImplementedError()

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`.
           Primarily used when a checkpoint is going to be evaluated without prior knowledge of its model configuration.
        """
        from compressai.zoo.image import cfgs
        config = cfgs['zyc2022-swint-hyperprior']  # TODO: self._config_from_state_dict should be called instead.
        net = cls(**config['M'])
        net.load_state_dict(state_dict)
        return net


class SwinTChARM(ConvChARM):
    """SwinT-ChARM
    Y. Zhu, Y. Yang, T. Cohen:
    "Transformer-Based Transform Coding"

    International Conference on Learning Representations (ICLR), 2022
    https://openreview.net/pdf?id=IDwN6xjHnK8
    """

    def __init__(self, *args, **kwargs):
        super().__init__(main_dim=kwargs['g_a']['embed_dim'][-1], hyper_dim=kwargs['h_a']['embed_dim'][-1])
        self.g_a = SwinTAnalysisTransform(**kwargs['g_a'])
        self.g_s = SwinTSynthesisTransform(**kwargs['g_s'])
        self.h_a = SwinTHyperAnalysisTransform(**kwargs['h_a'])
        self.h_s = SwinTHyperSynthesisTransform(**kwargs['h_s'])

    def _config_from_state_dict(self, state_dict):
        raise NotImplementedError()

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`.
           Primarily used when a checkpoint is going to be evaluated without prior knowledge of its model configuration.
        """
        from compressai.zoo.image import cfgs
        config = cfgs['zyc2022-swint-charm']  # TODO: self._config_from_state_dict should be called instead.
        net = cls(**config['M'])
        net.load_state_dict(state_dict)
        return net
