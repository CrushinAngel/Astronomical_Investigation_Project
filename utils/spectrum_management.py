
def resampling_spectra(filename):
    from astropy import units as u
    import numpy as np
    from astropy.visualization import quantity_support
    import msaexp
    quantity_support()  # for getting units on the axes below

    specs = msaexp.spectrum.get_spectra(
        file = filename, 
        sx=8, sy=2,
        ny=4, renorm = False
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

        resampled.append(resamp)

    return resampled



def weighted_average(resampled):
    from astropy import units as u
    import numpy as np
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
        meta = {'z': resampled[0].meta['z']}
    )
    # print(z)

    return combined_spectrum