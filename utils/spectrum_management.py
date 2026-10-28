
import matplotlib.pyplot as plt
from astropy import units as u
import numpy as np

def resampling_spectra(filename):

    from astropy.visualization import quantity_support
    import msaexp
    quantity_support()  # for getting units on the axes below

    specs = msaexp.spectrum.get_spectra(
        file = filename, 
        sx=8, sy=2,
        ny=4, renorm = False, 
        survey = 'AURORA'
    )


    delta_tryout = 0.0006000000000000449

    common_wave = (
        np.arange(0.55, 5.509904761905082 + delta_tryout, delta_tryout) * u.um
    )

    from specutils.manipulation import FluxConservingResampler
    resampler = FluxConservingResampler()

    resampled = []

    for spec in specs:

        resamp = resampler.resample1d(spec, common_wave)

        resamp.meta.update({
            "root": spec.meta["root"],
            "srcid": spec.meta["srcid"],
            "z": spec.meta["z"],
        })

        resampled.append(resamp)

    return resampled



def weighted_average(resampled):

    from specutils import Spectrum
    from astropy.nddata import InverseVariance, StdDevUncertainty

    fluxes = np.array([
            resamp.flux.value for resamp in resampled
    ])

    ivars = np.array([
            resamp.uncertainty.represent_as(InverseVariance).array for resamp in resampled
    ])

    valid = (
        np.isfinite(fluxes) &
        np.isfinite(ivars) &
        (fluxes != 0) & 
        (ivars > 0)
    )

    weights = np.where(valid, ivars, 0)
    flux_valid = np.where(valid, fluxes, 0)

    weight_sum = np.sum(weights, axis = 0)
    weighted_flux = np.full(fluxes.shape[1], np.nan)

    has_data = weight_sum > 0

    weighted_flux[has_data] = (
        np.sum(weights[:, has_data] * flux_valid[:, has_data], axis = 0)/weight_sum[has_data]
    )

    weighted_std = np.full_like(weighted_flux, np.nan)

    weighted_std[has_data] = 1 / np.sqrt(weight_sum[has_data])

    z = resampled[0].meta['z']

    combined_flux = weighted_flux * resampled[0].flux.unit
    combined_std = weighted_std * resampled[0].flux.unit
    rest_spectral_axis = resampled[0].spectral_axis/(1 + z)

    # target_unit = u.Unit('1e-20 erg / s / cm2 / AA')

    # combined_flux_flam = combined_flux.to(target_unit, equivalencies = u.spectral_density(rest_spectral_axis))
    # combined_std_flam = combined_std.to(target_unit, equivalencies = u.spectral_density(rest_spectral_axis))

    # print(combined_flux.unit)
    # print(combined_flux_flam.unit)

    uncertainty = StdDevUncertainty(combined_std)

    combined_spectrum = Spectrum(
        flux = combined_flux,
        spectral_axis = resampled[0].spectral_axis.value * 1e4 * u.AA,
        uncertainty = uncertainty, 
        meta = {"z": resampled[0].meta["z"],
                "root": resampled[0].meta["root"],
                "srcid": resampled[0].meta["srcid"],}
    )

    return combined_spectrum

def plot_spectrum(combined_spectrum, resampled):

    from pathlib import Path

    root = combined_spectrum.meta["root"]
    srcid = combined_spectrum.meta["srcid"]

    plot_dir = Path("..") / "figures" / f"{root}_{srcid}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    colors = ['mediumvioletred', 'lightseagreen', 'lightcoral']

    fig, axes = plt.subplots(3, 1, figsize = (10, 10))

    n = len(combined_spectrum.spectral_axis)
    cutoff1 = n // 3
    cutoff2 = 2 * n // 3

    sections = [
        slice(0, cutoff1),
        slice(cutoff1, cutoff2),
        slice(cutoff2, None)
    ]

    for ax, section in zip(axes, sections):

        ax.plot(
            combined_spectrum.spectral_axis[section],
            combined_spectrum.flux[section],
            c='midnightblue',
            label='Combined Spectrum'
        )

        for i, resample in enumerate(resampled):
            wave = resample.spectral_axis[section]
            flux = resample.flux[section]

            if np.any(np.isfinite(flux.value)):
                ax.plot(
                wave,
                flux,
                c=colors[i % len(colors)],
                label=f'Grating {i+1}',
                alpha = 0.3
                )
                
                max_flux, min_flux = np.nanmax(flux.value), np.nanmin(flux.value)

                if min_flux < -1.5:
                    ax.set_ylim(-1.5 * u.uJy, max_flux)

                elif max_flux > 20:
                    ax.set_ylim(min_flux, 20 * u.uJy)
                else:
                    ax.set_ylim(top = max_flux)

        ax.legend()

    fig.suptitle(f'Combined and individual spectrum visualization')

    plt.tight_layout()
    output_file = plot_dir / f"{root}_{srcid}_vis.png"

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    return