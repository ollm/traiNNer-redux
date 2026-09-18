import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class NoiseStatisticsLoss(nn.Module):
    """Compares image-noise statistics without matching noise pixel by pixel."""

    def __init__(
        self,
        loss_weight: float,
        blur_kernel_size: int = 5,
        patch_size: int = 16,
        area_stride: int | None = None,
        analysis_scales: tuple[int, ...] = (1, 2, 4),
        spectrum_bands: int = 8,
        max_lag: int = 4,
        level_weight: float = 1.0,
        local_weight: float = 1.0,
        spectrum_weight: float = 1.0,
        spectrum_energy_weight: float = 0.5,
        correlation_weight: float = 1.0,
        mean_weight: float = 0.25,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if blur_kernel_size < 3 or blur_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size must be an odd integer >= 3.")
        if patch_size < 2:
            raise ValueError("patch_size must be >= 2.")
        if area_stride is not None and area_stride < 1:
            raise ValueError("area_stride must be >= 1 when provided.")
        if not analysis_scales or any(scale < 1 for scale in analysis_scales):
            raise ValueError("analysis_scales must contain positive integers.")
        if len(set(analysis_scales)) != len(analysis_scales):
            raise ValueError("analysis_scales must not contain duplicates.")
        if spectrum_bands < 1:
            raise ValueError("spectrum_bands must be >= 1.")
        if max_lag < 1:
            raise ValueError("max_lag must be >= 1.")
        if min(
            level_weight,
            local_weight,
            spectrum_weight,
            spectrum_energy_weight,
            correlation_weight,
            mean_weight,
        ) < 0:
            raise ValueError("Noise statistic weights must be non-negative.")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.loss_weight = loss_weight
        self.patch_size = patch_size
        self.area_stride = area_stride or max(1, patch_size // 2)
        self.analysis_scales = analysis_scales
        self.spectrum_bands = spectrum_bands
        self.max_lag = max_lag
        self.level_weight = level_weight
        self.local_weight = local_weight
        self.spectrum_weight = spectrum_weight
        self.spectrum_energy_weight = spectrum_energy_weight
        self.correlation_weight = correlation_weight
        self.mean_weight = mean_weight
        self.eps = eps

        coordinates = torch.arange(blur_kernel_size, dtype=torch.float32)
        coordinates -= (blur_kernel_size - 1) / 2
        gaussian = torch.exp(-(coordinates**2) / (2 * (blur_kernel_size / 3) ** 2))
        gaussian = gaussian / gaussian.sum()
        kernel = gaussian[:, None] * gaussian[None, :]
        self.register_buffer("blur_kernel", kernel[None, None])

    def _validate_inputs(self, pred: Tensor, target: Tensor) -> None:
        if pred.ndim != 4 or target.ndim != 4:
            raise ValueError(
                "NoiseStatisticsLoss expects pred and target with shape [B, C, H, W]."
            )
        if pred.shape != target.shape:
            raise ValueError("NoiseStatisticsLoss expects matching pred and target shapes.")
        if pred.shape[1] < 1:
            raise ValueError("NoiseStatisticsLoss expects at least one channel.")
        if min(pred.shape[-2:]) < 2:
            raise ValueError("NoiseStatisticsLoss expects spatial dimensions >= 2.")

    def _extract_noise(self, image: Tensor) -> Tensor:
        channels = image.shape[1]
        kernel_size = min(self.blur_kernel.shape[-1], image.shape[-2], image.shape[-1])
        if kernel_size % 2 == 0:
            kernel_size -= 1
        kernel_start = (self.blur_kernel.shape[-1] - kernel_size) // 2
        kernel = self.blur_kernel[
            :, :, kernel_start : kernel_start + kernel_size, kernel_start : kernel_start + kernel_size
        ]
        kernel = kernel.to(dtype=image.dtype).expand(channels, 1, -1, -1)
        radius = kernel_size // 2
        padded = F.pad(image, (radius, radius, radius, radius), mode="reflect")
        smooth = F.conv2d(padded, kernel, groups=channels)
        return image - smooth

    def _local_mean(self, noise: Tensor) -> Tensor:
        kernel_h = min(self.patch_size, noise.shape[-2])
        kernel_w = min(self.patch_size, noise.shape[-1])
        stride_h = min(self.area_stride, kernel_h)
        stride_w = min(self.area_stride, kernel_w)
        padding_h = kernel_h // 2
        padding_w = kernel_w // 2
        return F.avg_pool2d(
            noise,
            (kernel_h, kernel_w),
            (stride_h, stride_w),
            (padding_h, padding_w),
            count_include_pad=False,
        )

    def _local_std(self, noise: Tensor) -> Tensor:
        kernel_h = min(self.patch_size, noise.shape[-2])
        kernel_w = min(self.patch_size, noise.shape[-1])
        stride_h = min(self.area_stride, kernel_h)
        stride_w = min(self.area_stride, kernel_w)
        padding_h = kernel_h // 2
        padding_w = kernel_w // 2
        mean = F.avg_pool2d(
            noise,
            (kernel_h, kernel_w),
            (stride_h, stride_w),
            (padding_h, padding_w),
            count_include_pad=False,
        )
        mean_square = F.avg_pool2d(
            noise.square(),
            (kernel_h, kernel_w),
            (stride_h, stride_w),
            (padding_h, padding_w),
            count_include_pad=False,
        )
        return (mean_square - mean.square()).clamp_min(self.eps).sqrt()

    def _local_rms(self, noise: Tensor) -> Tensor:
        kernel_h = min(self.patch_size, noise.shape[-2])
        kernel_w = min(self.patch_size, noise.shape[-1])
        stride_h = min(self.area_stride, kernel_h)
        stride_w = min(self.area_stride, kernel_w)
        padding_h = kernel_h // 2
        padding_w = kernel_w // 2
        return F.avg_pool2d(
            noise.square(),
            (kernel_h, kernel_w),
            (stride_h, stride_w),
            (padding_h, padding_w),
            count_include_pad=False,
        ).clamp_min(self.eps).sqrt()

    def _spectrum_statistics(self, noise: Tensor) -> tuple[Tensor, Tensor]:
        height, width = noise.shape[-2:]
        spectrum = torch.fft.rfft2(noise, norm="ortho").abs().square()
        frequencies_y = torch.fft.fftfreq(height, device=noise.device).abs()
        frequencies_x = torch.fft.rfftfreq(width, device=noise.device)
        radius = torch.sqrt(
            frequencies_y[:, None].square() + frequencies_x[None, :].square()
        )
        edges = torch.linspace(
            0,
            radius.detach().amax().clamp_min(self.eps),
            self.spectrum_bands + 1,
            device=noise.device,
            dtype=radius.dtype,
        )
        band_powers = []
        for index in range(self.spectrum_bands):
            band = (radius >= edges[index]) & (radius < edges[index + 1])
            if band.any():
                band_power = spectrum[..., band].mean(dim=-1)
            else:
                band_power = spectrum.new_zeros(spectrum.shape[:-2])
            band_powers.append(band_power)
        powers = torch.stack(band_powers, dim=-1)
        profile = powers / (powers.sum(dim=-1, keepdim=True) + self.eps)
        energy = torch.log1p(powers)
        return profile, energy

    def _autocorrelation(self, noise: Tensor) -> Tensor:
        centered = noise - noise.mean(dim=(-2, -1), keepdim=True)
        rms = (
            centered.square()
            .mean(dim=(-2, -1), keepdim=True)
            .clamp_min(self.eps)
            .sqrt()
        )
        correlations = []
        max_vertical_lag = min(self.max_lag, centered.shape[-2] - 1)
        max_horizontal_lag = min(self.max_lag, centered.shape[-1] - 1)
        for lag in range(1, self.max_lag + 1):
            if lag <= max_vertical_lag:
                vertical = (centered[..., lag:, :] * centered[..., :-lag, :]).mean(
                    dim=(-2, -1), keepdim=True
                )
                vertical = vertical / (rms[..., :, :] * rms[..., :, :] + self.eps)
            else:
                vertical = centered.new_zeros(centered.shape[0], centered.shape[1], 1, 1)
            if lag <= max_horizontal_lag:
                horizontal = (centered[..., :, lag:] * centered[..., :, :-lag]).mean(
                    dim=(-2, -1), keepdim=True
                )
                horizontal = horizontal / (rms[..., :, :] * rms[..., :, :] + self.eps)
            else:
                horizontal = centered.new_zeros(centered.shape[0], centered.shape[1], 1, 1)
            correlations.extend((vertical.squeeze(-1).squeeze(-1), horizontal.squeeze(-1).squeeze(-1)))
        return torch.stack(correlations, dim=-1)

    def _single_scale_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        pred_noise = self._extract_noise(pred)
        target_noise = self._extract_noise(target)

        pred_rms = pred_noise.square().mean(dim=(-2, -1)).clamp_min(self.eps).sqrt()
        target_rms = target_noise.square().mean(dim=(-2, -1)).clamp_min(self.eps).sqrt()
        level_loss = F.l1_loss(torch.log1p(pred_rms), torch.log1p(target_rms))

        local_loss = F.l1_loss(
            torch.log1p(self._local_std(pred_noise)),
            torch.log1p(self._local_std(target_noise)),
        )
        area_level_loss = F.l1_loss(
            torch.log1p(self._local_rms(pred_noise)),
            torch.log1p(self._local_rms(target_noise)),
        )
        pred_spectrum, pred_spectrum_energy = self._spectrum_statistics(pred_noise)
        target_spectrum, target_spectrum_energy = self._spectrum_statistics(
            target_noise
        )
        spectrum_loss = F.l1_loss(pred_spectrum, target_spectrum)
        spectrum_energy_loss = F.l1_loss(
            pred_spectrum_energy, target_spectrum_energy
        )
        correlation_loss = F.l1_loss(
            self._autocorrelation(pred_noise), self._autocorrelation(target_noise)
        )
        mean_loss = F.l1_loss(
            self._local_mean(pred_noise), self._local_mean(target_noise)
        )

        return (
            self.level_weight * level_loss
            + self.local_weight * (local_loss + area_level_loss)
            + self.spectrum_weight * spectrum_loss
            + self.spectrum_energy_weight * spectrum_energy_loss
            + self.correlation_weight * correlation_loss
            + self.mean_weight * mean_loss
        )

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        self._validate_inputs(pred, target)
        # Squared statistics and FFTs are numerically unstable in AMP dtypes.
        pred = pred.float()
        target = target.float()
        height, width = pred.shape[-2:]
        scale_losses = []
        for scale in self.analysis_scales:
            if min(height, width) < scale * 2:
                continue
            if scale == 1:
                pred_analysis = pred
                target_analysis = target
            else:
                pred_analysis = F.avg_pool2d(pred, scale, stride=scale)
                target_analysis = F.avg_pool2d(target, scale, stride=scale)
            scale_losses.append(self._single_scale_loss(pred_analysis, target_analysis))

        if not scale_losses:
            raise ValueError("No analysis scale fits the input spatial dimensions.")
        return torch.stack(scale_losses).mean()


@LOSS_REGISTRY.register()
def noise_statistics(**kwargs: object) -> NoiseStatisticsLoss:
    return NoiseStatisticsLoss(**kwargs)  # type: ignore[arg-type]
