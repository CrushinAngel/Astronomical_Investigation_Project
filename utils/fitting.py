def gauss(x, mean, sigma, flux):
    gaussian = flux / (sigma * np.sqrt(2 * np.pi)) * np.exp((-(x - mean)**2) / (2* sigma**2))
    return gaussian


def model1(x, a, b, sigma,
            mean, flux):

    cont = a + b*x

    return (cont + gauss(x, mean, sigma, flux))

def model2(x, a, b, sigma,
            mean1, flux1, mean2, flux2):

    cont = a + b*x

    return (cont 
            + gauss(x, mean1, sigma, flux1)
            + gauss(x, mean2, sigma, flux2))

def model3(x, a, b, sigma,
            mean1, flux1, mean2, flux2, mean3, flux3):

    cont = a + b*x

    return (cont 
            + gauss(x, mean1, sigma, flux1)
            + gauss(x, mean2, sigma, flux2)
            + gauss(x, mean3, sigma, flux3))

def line_cont_fit(spectrum, resampled, line_info):
    from astropy import units as u
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from astropy.nddata import StdDevUncertainty

    from pathlib import Path

    root = spectrum.meta["root"]
    srcid = spectrum.meta["srcid"]

    plot_dir = Path("..") / "figures" / f"{root}_{srcid}"
    plot_dir.mkdir(parents=True, exist_ok=True)
                

    z = spectrum.meta['z']

    print(*line_info['element'])

    ###############################################################################################

    ## Centering + masking (maybe needs change)

    center, left_win, right_win = line_info['mask_info']
    center_obs = center * (1+z)

    mask = ((spectrum.spectral_axis >= left_win * (1+z)) & 
            (spectrum.spectral_axis <= right_win * (1+z)))

    wave_masked = spectrum.spectral_axis[mask]
    flux_masked = spectrum.flux[mask]
    # flux_err_masked = spectrum.uncertainty.represent_as(StdDevUncertainty).array[mask]

    file = open(plot_dir / f'{line_info['element'][0]}_log.txt', 'w')
    file.write(f"Line: {line_info['element'][0]} \n")
    file.write(f"Redshift: {z} \n")
    file.write(f"Rest window: {left_win} -- {right_win} \n")
    file.write(f"Observed window: {left_win * (1 + z)} -- {right_win * (1 + z)} \n")

    file.write(f"Spectrum range: {spectrum.spectral_axis.min()} -- "
        f"{spectrum.spectral_axis.max()} \n")

    file.write(f"Points selected: {len(wave_masked)} \n")

    if len(wave_masked) > 0:
        file.write(f"Selected wavelength range: "
            f"{wave_masked.min()} -- {wave_masked.max()} \n")

        file.write(f"Finite wavelengths: "
            f"{np.isfinite(wave_masked.value).sum()} \n")

        file.write(f"Finite fluxes: "
            f"{np.isfinite(flux_masked.value).sum()} \n")


    
    flux_masked = flux_masked.to(u.erg / u.cm**2) / (wave_masked**2) * (3*1e18 * (u.AA / u.s))  
    flux_err_masked = spectrum.uncertainty.quantity[mask].to(u.erg / u.cm**2) / (wave_masked**2) * (3*1e18 * (u.AA / u.s))  
    flux_err_masked = StdDevUncertainty(flux_err_masked).array


    # Validity mask
    valid = (
        np.isfinite(flux_masked.value) &
        np.isfinite(flux_err_masked) &
        (flux_err_masked > 0)
    )

    wave_valid = wave_masked[valid]
    flux_masked = flux_masked[valid]
    flux_err_masked = flux_err_masked[valid]

    # Keep the individual grating spectra in the fitting window
    grating_spectra = []

    for i, resample in enumerate(resampled):
        spec_wave = resample.spectral_axis.to(wave_masked.unit)
        spec_flux = resample.flux

        # Select the same observed-frame window as the averaged spectrum
        spec_window = (
            (spec_wave >= left_win * (1 + z)) &
            (spec_wave <= right_win * (1 + z))
        )

        if not np.any(spec_window):
            continue

        # Convert to the same flux units used for the averaged spectrum
        spec_flux = (
            spec_flux.to(u.erg / u.cm**2)
            / spec_wave**2
            * (3e18 * (u.AA / u.s))
        )

        # Exclude invalid samples
        valid_flux = (
            spec_window &
            np.isfinite(spec_flux.value)
        )

        if not np.any(valid_flux):
            continue

        grating = resample.meta.get("grating", f"Grating {i + 1}")
        filt = resample.meta.get("filter", "Unknown filter")

        grating_spectra.append((
            spec_wave[valid_flux],
            spec_flux[valid_flux],
            f"{grating} / {filt}"
        ))

        file.write(
            f"{grating} / {filt}: "
            f"{np.count_nonzero(valid_flux)} finite samples \n"
        )

    if len(flux_masked) == 0:
        file.write(
            f"Skipping {line_info['element'][0]}: "
            "no valid flux measurements in the fitting window. \n"
        )
        return {}, {}


    def max_amplitude(rest_AA):
        line_mask = ((wave_valid >= rest_AA*(1+z) - 20*u.AA) & 
                     (wave_valid <= rest_AA*(1+z) + 20*u.AA))
        line_profile = flux_masked[line_mask]
        wave_lims = wave_valid[line_mask]

        if len(line_profile) < 5:
            file.write(f"No valid samples around {rest_AA} Å. \n")
            return [line_mask, np.nan, np.nan]
    
        max_idx = np.argmax(line_profile)
        max_val = line_profile[max_idx].value
        max_wave = wave_lims[max_idx]
        return [line_mask, max_val, max_wave]
    
    wave_model = np.linspace(
        wave_valid.value.min(),
        wave_valid.value.max(),
        5000
    )

    colors = ['mediumvioletred', 'lightseagreen', 'lightcoral']

    ###############################################################################################

    ## Curvefit with the different models (1, 2, or 3 gaussians)
    ## Optimal values and errors for each amount of gaussian case
    ## important to consider which are the 1st, 2nd and 3rd elements in a list
    
    if line_info['lines'] == 1:

        line_mask, max_amp, max_wave = max_amplitude(line_info['rest_AA'])

        if not np.isfinite(max_amp) or not np.isfinite(max_wave):
            file.write(f"Skipping {line_info['element'][0]}: cannot initialize line fit. \n")
            return {}, {}

        initial = [np.median(flux_masked.value), # The median of the flux as a guess for the intercept
                    0,                            # The slope is practically null for the continuum
                    line_info['sigma'],                          # Sigma, assuming most of the lines aren't broad components
                    line_info['rest_AA'].value * (1 + z),              # Rest in AA * (redshift correction since the wave array is observed)
                    line_info['sigma'] * np.sqrt(2*np.pi) * max_amp]       # Then the expected flux, not amplitude

        popt, pcov = curve_fit(model1, wave_valid.value, flux_masked.value, p0 = initial, 
                        sigma = flux_err_masked, absolute_sigma = True)

        flux_model = model1(wave_model, *popt)

        intercept, slope, sigma, mean, flux = popt

        model_errs = np.sqrt(np.diag(pcov))
        intercept_err, slope_err, sigma_err, mean_err, flux_err = model_errs

        snr = np.abs(flux/flux_err)
        file.write(f'SNR: {snr}\n')

        if snr < 3:
            fixed = [sigma, mean]
            flux, flux_err = refit_flux1(wave_valid.value, flux_masked.value, 
                              initial, flux_err_masked, fixed)
            detection = False
            file.write(f'Upper limit flux: {flux}\n'
                       f'Upper limit flux_err: {flux_err}\n')
        else: detection = True

        optimal_dict = {
            'element': line_info['element'],
            'lines': line_info['lines'],
            'detection': detection,
            'intercept': intercept, 
            'slope': slope, 
            'sigma': sigma, 
            'mean1': mean,
            'flux1': flux,
        }

        opterrs_dict = {
            'intercept': intercept_err, 
            'slope': slope_err, 
            'sigma': sigma_err, 
            'mean1': mean_err,
            'flux1': flux_err,
        }

        ###############################################################################################

        ## Plotting!

        f, (ax, chi) = plt.subplots(2, 1, 
                                    figsize = (10, 6), 
                                    sharex = True, 
                                    gridspec_kw={'height_ratios': [2, 1], 
                                                 'hspace': 0.3})

        ax.xaxis.set_visible(True)

        flux = flux
        mu = mean
        textstr = '\n'.join((
            r'$Intercept =%.2e$' % (intercept, ), 
            r'$Slope =%.2e$' % (slope, ),
            r'$\sigma =%.2f$' % (sigma, ),
            r'$\mu=%.2f$' % (mu, ),
            r'$F=%.2e$' % (flux, )))

        ax.plot(wave_valid, flux_masked, color = 'midnightblue', label = 'Averaged Spectrum')

        plt.autoscale(enable=False, axis='y')

        if len(grating_spectra) > 1:
            for spec_wave, spec_flux, label in grating_spectra:
                ax.plot(
                    spec_wave.value,
                    spec_flux.value,
                    marker = "d",
                    markersize = 1.5,
                    linewidth = 0.8,
                    alpha = 0.5,
                    label=label
                )

        ax.plot(wave_model, flux_model, color = 'darkred', alpha = 0.5, label = 'Best Fit')
        ax.scatter(max_wave, max_amp, c = colors[0])
        ax.fill_between(wave_valid, ax.get_ylim()[0], ax.get_ylim()[1], 
                where = line_mask, color = colors[0], alpha = 0.1,
                label = "Masked Region")

        ax.axvline(center_obs, c = 'crimson', label = 'Line centroid')
        ax.minorticks_on()

        props = dict(boxstyle='round', facecolor='white', alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=props, c = 'midnightblue')
    
        # ax.set_xlim(0.48, 0.51)
        # ax.set_ylim(-1, 20)
        ax.set_title(f'Isolated {line_info['element'][0]} fitting')
        ax.set_xlabel(wave_valid.unit.to_string())
        ax.grid(alpha = 0.5)
        ax.legend(loc='upper right')

        ax.tick_params(axis='x', labelbottom=True)

        flux_fit_points = model1(wave_valid.value, *popt)

        residuals = flux_masked.value - flux_fit_points

        normalized_residuals = residuals / flux_err_masked

        chi_squared = np.sum(normalized_residuals**2)

        dof = len(normalized_residuals) - len(popt)
        reduced_chi_squared = chi_squared / dof

        chi.scatter(wave_valid, normalized_residuals, s=12)

        chi.axhline(0, color='black', linewidth=1)
        chi.axhline(1, color='gray', linestyle='--')
        chi.axhline(-1, color='gray', linestyle='--')

        chi.set_xlabel('')
        chi.set_ylabel('Normalized Residuals')
        chi.tick_params(axis='x', labelbottom=False)
        chi.grid(alpha = 0.5)

        f.savefig(
            plot_dir / f"{line_info['element'][0]}_fit.png",
            dpi=200,
            bbox_inches="tight"
        )
        plt.close(f)
        file.close()

        ###############################################################################################

    if line_info['lines'] == 2:

        line_mask1, max_amp1, max_wave1 = max_amplitude(line_info['rest_AA'][0])
        line_mask2, max_amp2, max_wave2 = max_amplitude(line_info['rest_AA'][1])

        if not np.isfinite(max_amp1) or not np.isfinite(max_wave1) or not np.isfinite(max_amp2) or not np.isfinite(max_wave2):
            file.write(f"Skipping {line_info['element'][0]}: cannot initialize line fit.")
            return {}, {}

        initial = [np.median(flux_masked.value), 
                    0,                           
                    line_info['sigma'],
                    line_info['rest_AA'][0].value * (1 + z),
                    line_info['sigma'] * np.sqrt(2*np.pi) * max_amp1,
                    line_info['rest_AA'][1].value * (1 + z),
                    line_info['sigma'] * np.sqrt(2*np.pi) * max_amp2]

        popt, pcov = curve_fit(model2, wave_valid.value, flux_masked.value, p0 = initial, 
                        sigma = flux_err_masked, absolute_sigma = True)

        flux_model = model2(wave_model, *popt)

        intercept, slope, sigma, mean1, flux1, mean2, flux2 = popt

        model_errs = np.sqrt(np.diag(pcov))
        intercept_err, slope_err, sigma_err, mean_err1, flux_err1, mean_err2, flux_err2 = model_errs

        snr1 = np.abs(flux1/flux_err1)
        snr2 = np.abs(flux2/flux_err2)

        
        file.write(f'SNR1: {snr1}\n'
                   f'SNR2: {snr2} \n')
        detection = np.zeros(2, dtype = bool)

        if snr1 < 3:
            fixed = [sigma, mean1]
            flux1, flux_err1 = refit_flux2(wave_valid.value, flux_masked.value, 
                                         initial, flux_err_masked, fixed, line = 1)
            detection[0] = False
            file.write(f'Upper limit flux1: {flux1}\n'
                       f'Upper limit flux_err1: {flux_err1}\n')
        else: detection[0] = True

        if snr2 < 3:
            fixed = [sigma, mean2]
            flux2, flux_err2 = refit_flux2(wave_valid.value, flux_masked.value, 
                                         initial, flux_err_masked, fixed, line = 2)
            detection[1] = False
            file.write(f'Upper limit flux2: {flux2}\n'
                       f'Upper limit flux_err2: {flux_err2}\n')
        else: detection[1] = True

        optimal_dict = {
            'element': line_info['element'],
            'lines': line_info['lines'],
            'detection': detection,
            'intercept': intercept, 
            'slope': slope, 
            'sigma': sigma, 
            'mean1': mean1,
            'flux1': flux1,
            'mean2': mean2,
            'flux2': flux2,
        }

        opterrs_dict = {
            'intercept': intercept_err, 
            'slope': slope_err, 
            'sigma': sigma_err, 
            'mean1': mean_err1,
            'flux1': flux_err1,
            'mean2': mean_err2,
            'flux2': flux_err2,
        }

        ###############################################################################################

        ## Plotting!

        f, (ax, chi) = plt.subplots(2, 1, 
                                    figsize = (10, 6), 
                                    sharex = True, 
                                    gridspec_kw={'height_ratios': [2, 1], 
                                                 'hspace': 0.3})

        ax.xaxis.set_visible(True)

        mu1 = mean1
        flux1 = flux1

        mu2 = mean2
        flux2 = flux2

        textstr = '\n'.join((
            r'$Intercept =%.2e$' % (intercept, ), 
            r'$Slope =%.2e$' % (slope, ),
            r'$\sigma =%.2f$' % (sigma, ),
            r'$\mu1=%.2f$' % (mu1, ),
            r'$F 1=%.2e$' % (flux1, ),
            r'$\mu2=%.2f$' % (mu2, ),
            r'$F 2=%.2e$' % (flux2, )))

        ax.plot(wave_valid, flux_masked, color = 'midnightblue', label = 'Averaged Spectrum')

        plt.autoscale(enable=False, axis='y')

        if len(grating_spectra) > 1:
            for spec_wave, spec_flux, label in grating_spectra:
                ax.plot(
                    spec_wave.value,
                    spec_flux.value,
                    marker = "d",
                    markersize = 1.5,
                    linewidth = 0.8,
                    alpha = 0.5,
                    label=label
                )

        ax.plot(wave_model, flux_model, color = 'darkred', alpha = 0.5, label = 'Best Fit')

        ax.scatter([max_wave1, max_wave2], [max_amp1, max_amp2], c = [colors[0], colors[1]])
        ax.fill_between(wave_valid, ax.get_ylim()[0], ax.get_ylim()[1], 
                        where = line_mask1, color = colors[0], alpha = 0.1,
                        label = "Masked Region 1")

        ax.fill_between(wave_valid, ax.get_ylim()[0], ax.get_ylim()[1], 
                        where = line_mask2, color = colors[1], alpha = 0.1, 
                        label = "Masked Region 2")

        ax.axvline(center_obs, c = 'crimson', label = 'Line centroid')
        
        ax.minorticks_on()

        props = dict(boxstyle='round', facecolor='white', alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=props, c = 'midnightblue')
    
        # ax.set_xlim(0.48, 0.51)
        # ax.set_ylim(-1, 20)
        ax.set_title(f'Isolated {line_info['element'][0]}, {line_info['element'][1]} fitting')
        ax.set_xlabel(wave_valid.unit.to_string())
        ax.grid(alpha = 0.5)
        ax.legend(loc='upper right')

        ax.tick_params(axis='x', labelbottom=True)

        flux_fit_points = model2(wave_valid.value, *popt)

        residuals = flux_masked.value - flux_fit_points

        normalized_residuals = residuals / flux_err_masked

        chi_squared = np.sum(normalized_residuals**2)

        dof = len(normalized_residuals) - len(popt)
        reduced_chi_squared = chi_squared / dof

        chi.scatter(wave_valid, normalized_residuals, s=12)

        chi.axhline(0, color='black', linewidth=1)
        chi.axhline(1, color='gray', linestyle='--')
        chi.axhline(-1, color='gray', linestyle='--')

        chi.set_xlabel('')
        chi.set_ylabel('Normalized Residuals')
        chi.tick_params(axis='x', labelbottom=False)
        chi.grid(alpha = 0.5)

        f.savefig(
            plot_dir / f"{line_info['element'][0]}_{line_info['element'][1]}fit.png",
            dpi=200,
            bbox_inches="tight"
        )
        plt.close(f)
        file.close()

        ###############################################################################################

    if line_info['lines'] == 3:

        line_mask1, max_amp1, max_wave1 = max_amplitude(line_info['rest_AA'][0])
        line_mask2, max_amp2, max_wave2 = max_amplitude(line_info['rest_AA'][1])
        line_mask3, max_amp3, max_wave3 = max_amplitude(line_info['rest_AA'][2])

        if not np.isfinite(max_amp1) or not np.isfinite(max_wave1) or not np.isfinite(max_amp2) or not np.isfinite(max_wave2) or not np.isfinite(max_amp3) or not np.isfinite(max_wave3):
            file.write(f"Skipping {line_info['element'][0]}: cannot initialize line fit.")
            return {}, {}

        initial = [np.median(flux_masked.value), 
                    0,                           
                    line_info['sigma'],
                    line_info['rest_AA'][0].value * (1 + z),
                    line_info['sigma'] * np.sqrt(2*np.pi) * max_amp1,
                    line_info['rest_AA'][1].value * (1 + z),
                    line_info['sigma'] * np.sqrt(2*np.pi) * max_amp2, 
                    line_info['rest_AA'][2].value * (1 + z),
                    line_info['sigma'] * np.sqrt(2*np.pi) * max_amp3]


        popt, pcov = curve_fit(model3, wave_valid.value, flux_masked.value, p0 = initial, 
                                sigma = flux_err_masked, absolute_sigma = True)
        
        flux_model = model3(wave_model, *popt)

        intercept, slope, sigma, mean1, flux1, mean2, flux2, mean3, flux3 = popt

        model_errs = np.sqrt(np.diag(pcov))
        intercept_err, slope_err, sigma_err, mean_err1, flux_err1, mean_err2, flux_err2, mean_err3, flux_err3 = model_errs

        snr1 = np.abs(flux1/flux_err1)
        snr2 = np.abs(flux2/flux_err2)
        snr3 = np.abs(flux3/flux_err3)
        file.write(f'SNR1: {snr1}\n' 
                   f'SNR2: {snr2}\n' 
                   f'SNR2: {snr3} \n')
        detection = np.zeros(3, dtype = bool)

        if snr1 < 3:
            fixed = [sigma, mean1]
            flux1, flux_err1 = refit_flux3(wave_valid.value, flux_masked.value, 
                              initial, flux_err_masked, fixed, line = 1)
            detection[0] = False
            file.write(f'Upper limit flux1: {flux1}\n'
                       f'Upper limit flux_err1: {flux_err1}\n')
        else: detection[0] = True

        if snr2 < 3:
            fixed = [sigma, mean2]
            flux2, flux_err2 = refit_flux3(wave_valid.value, flux_masked.value, 
                              initial, flux_err_masked, fixed, line = 2)
            detection[1] = False
            file.write(f'Upper limit flux2: {flux2}\n'
                       f'Upper limit flux_err2: {flux_err2}\n')
        else: detection[1] = True

        if snr3 < 3:
            fixed = [sigma, mean3]
            flux3, flux_err3 = refit_flux3(wave_valid.value, flux_masked.value, 
                              initial, flux_err_masked, fixed, line = 3)
            detection[2] = False
            file.write(f'Upper limit flux3: {flux3}\n'
                       f'Upper limit flux_err3: {flux_err3}\n')
        else: detection[2] = True
    
        optimal_dict = {
            'element': line_info['element'],
            'lines': line_info['lines'],
            'detection': detection,
            'intercept': intercept, 
            'slope': slope, 
            'sigma': sigma, 
            'mean1': mean1,
            'flux1': flux1,
            'mean2': mean2,
            'flux2': flux2,
            'mean3': mean3,
            'flux3': flux3,
        }

        opterrs_dict = {
            'intercept': intercept_err, 
            'slope': slope_err, 
            'sigma': sigma_err, 
            'mean1': mean_err1,
            'flux1': flux_err1,
            'mean2': mean_err2,
            'flux2': flux_err2,
            'mean3': mean_err3,
            'flux3': flux_err3,
        }    

        ###############################################################################################

        ## Plotting!

        f, (ax, chi) = plt.subplots(2, 1, 
                                    figsize = (10, 6), 
                                    sharex = True, 
                                    gridspec_kw={'height_ratios': [2, 1], 
                                                 'hspace': 0.3})

        ax.xaxis.set_visible(True)

        mu1 = mean1
        flux1 = flux1

        mu2 = mean2
        flux2 = flux2

        mu3 = mean3
        flux3 = flux3

        textstr = '\n'.join((
            r'$Intercept =%.2e$' % (intercept, ), 
            r'$Slope =%.2e$' % (slope, ),
            r'$\sigma =%.2f$' % (sigma, ),
            r'$\mu 1=%.2f$' % (mu1, ),
            r'$F 1=%.2e$' % (flux1, ),
            r'$\mu 2=%.2f$' % (mu2, ),
            r'$F 2=%.2e$' % (flux2, ),
            r'$\mu 3=%.2f$' % (mu3, ),
            r'$F 3=%.2e$' % (flux3, )))

        ax.plot(wave_valid, flux_masked, color = 'midnightblue', label = 'Averaged Spectrum')

        plt.autoscale(enable=False, axis='y')

        if len(grating_spectra) > 1:
            for spec_wave, spec_flux, label in grating_spectra:
                ax.plot(
                    spec_wave.value,
                    spec_flux.value,
                    marker = "d",
                    markersize = 1.5,
                    linewidth = 0.8,
                    alpha = 0.5,
                    label=label
                )

        ax.plot(wave_model, flux_model, color = 'darkred', alpha = 0.5, label = 'Best Fit')

        ax.scatter([max_wave1, max_wave2, max_wave3], [max_amp1, max_amp2, max_amp3], c = [colors[0], colors[1], colors[2]])
        ax.fill_between(wave_valid, ax.get_ylim()[0], ax.get_ylim()[1], 
                        where = line_mask1, color = colors[0], alpha = 0.1,
                        label = "Masked Region 1")

        ax.fill_between(wave_valid, ax.get_ylim()[0], ax.get_ylim()[1], 
                        where = line_mask2, color = colors[1], alpha = 0.1,
                        label = "Masked Region 2")

        ax.fill_between(wave_valid, ax.get_ylim()[0], ax.get_ylim()[1], 
                        where = line_mask3, color = colors[2], alpha = 0.1, 
                        label = "Masked Region 3")        

        ax.axvline(center_obs, c = 'crimson', label = 'Line centroid')

        ax.minorticks_on()

        props = dict(boxstyle='round', facecolor='white', alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=props, c = 'midnightblue')
    
        # ax.set_xlim(0.48, 0.51)
        # ax.set_ylim(-1, 20)
        ax.set_title(f'Isolated {line_info['element'][0]}, {line_info['element'][1]}, {line_info['element'][2]} fitting')
        ax.set_xlabel(wave_valid.unit.to_string())
        ax.grid(alpha = 0.5)
        ax.legend(loc='upper right')

        ax.tick_params(axis='x', labelbottom=True)

        flux_fit_points = model3(wave_valid.value, *popt)

        residuals = flux_masked.value - flux_fit_points

        normalized_residuals = residuals / flux_err_masked

        chi_squared = np.sum(normalized_residuals**2)

        dof = len(normalized_residuals) - len(popt)
        reduced_chi_squared = chi_squared / dof

        chi.scatter(wave_valid, normalized_residuals, s=12)

        chi.axhline(0, color='black', linewidth=1)
        chi.axhline(1, color='gray', linestyle='--')
        chi.axhline(-1, color='gray', linestyle='--')

        chi.set_xlabel('')
        chi.set_ylabel('Normalized Residuals')
        chi.tick_params(axis='x', labelbottom=False)
        chi.grid(alpha = 0.5)

        f.savefig(
            plot_dir / f"{line_info['element'][0]}_{line_info['element'][1]}{line_info['element'][2]}fit.png",
            dpi=200,
            bbox_inches="tight"
        )
        plt.close(f)
        file.close()

        ###############################################################################################


    return [optimal_dict, opterrs_dict]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import astropy.units as u
import numpy as np

import os
import sys

# Move up one directory level from /notebooks to the workspace root
workspace_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

if workspace_root not in sys.path:
    sys.path.append(workspace_root)
from utils.spectrum_management import (resampling_spectra, weighted_average, plot_spectrum)

Halpha = {
    'lines': 3,
    'element': ['Halpha', '[NII]6548', '[NII]6583'],
    'rest_AA': [6564 * u.AA, 6548 * u.AA, 6583 * u.AA], 
    'sigma': 7.5, 
    'mask_info': [6564 * u.AA, 6461.06 * u.AA, 6661.4 * u.AA]
}

Sii = {
    'lines': 2,
    'element': ['[SII]6717', '[SII]6731'],
    'rest_AA': [6717 * u.AA, 6731 * u.AA], 
    'sigma': 7.5, 
    'mask_info': [6724 * u.AA, 6602.58 * u.AA, 6877.68 * u.AA]
}

Oiii = {
    'lines': 2,
    'element': ['Hgamma', '[OIII]4364'],
    'rest_AA': [4340 * u.AA, 4364 * u.AA], 
    'sigma': 7.5, 
    'mask_info': [4354 * u.AA, 4264.16 * u.AA, 4424.64 * u.AA]
}

Oiii1 = {
    'lines': 1,
    'element': ['[OIII]4959'],
    'rest_AA': 4959 * u.AA, 
    'sigma': 7.5, 
    'mask_info': [4959 * u.AA, 4908.4 * u.AA, 4993.54 * u.AA] 
}

Oiii2 = {
    'lines': 1,
    'element': ['[OIII]5007'],
    'rest_AA': 5007 * u.AA, 
    'sigma': 7.5, 
    'mask_info': [5007 * u.AA, 4974.76 * u.AA, 5058.65 * u.AA] 
}

Hbeta = {
    'lines': 1,
    'element': ['Hbeta'],
    'rest_AA': 4861 * u.AA, 
    'sigma': 7.5, 
    'mask_info': [4861 * u.AA, 4783.18 * u.AA, 4933.44 * u.AA] 
}

Oii = {
    'lines': 1,
    'element': ['[OII]3727'],
    'rest_AA': 3727 * u.AA, 
    'sigma': 7.5, 
    'mask_info': [3727 * u.AA, 3576.39 * u.AA, 3851.5 * u.AA] 
}

Oiid = {
    'lines': 2,
    'element': ['[OII]7322', '[OII]7332'],
    'rest_AA': [7322 * u.AA, 7332 * u.AA], 
    'sigma': 7.5, 
    'mask_info': [7327 * u.AA, 7152.79 * u.AA, 7450.82 * u.AA] 
}

Oi = {
    'lines': 1,
    'element': ['[OI]6300'],
    'rest_AA': 6300 * u.AA, 
    'sigma': 7.5, 
    'mask_info': [6300 * u.AA, 6052.36 * u.AA, 6510.87 * u.AA] 
}

lines = [Halpha, Hbeta, Oi, Oii, Oiid, Oiii, Oiii1, Oiii2, Sii]

def refit_flux1(wave, flux, initial, fluxerr, fixed):
    from scipy.optimize import curve_fit

    sigma, mean = fixed
    intercept0, slope0, _, _, flux0 = initial
    intial_fixed = [intercept0, slope0, flux0]

    rng = np.random.default_rng(13)

    n_it = 100

    flux_mc = []
    flux_err_mc = []

    for it in range(n_it):

        flux_per = flux + rng.normal(0, fluxerr)

        try:
            popt, pcov = curve_fit(
                lambda x, intercept, slope, flux: model1(x, intercept, slope, 
                                                         sigma, mean, flux),
                wave,
                flux_per,
                p0 = intial_fixed,
                sigma = fluxerr,
                absolute_sigma = True
            )

            intercept_mc, slope_mc, flux_mc_i = popt
            err = np.sqrt(np.diag(pcov))
            flux_err_i = err[2]

            flux_mc.append(flux_mc_i)
            flux_err_mc.append(flux_err_i)

        except (RuntimeError, ValueError):
            continue

    flux_mc = np.array(flux_mc)
    flux_err_mc = np.array(flux_err_mc)

    upper_flux = 3 * np.std(flux_mc, ddof = 1)
    upper_flux_err = np.median(flux_err_mc)

    return upper_flux, upper_flux_err

def refit_flux2(wave, flux, initial, fluxerr, fixed, line):
    from scipy.optimize import curve_fit

    sigma, mean = fixed
    if line == 1:
        intercept0, slope0, _, _, flux10, mean20, flux20 = initial
        intial_fixed = [intercept0, slope0, flux10, mean20, flux20]
    else:
        intercept0, slope0, _, mean10, flux10, _, flux20 = initial
        intial_fixed = [intercept0, slope0, mean10, flux10, flux20]

    rng = np.random.default_rng(13)

    n_it = 100

    flux_mc = []
    flux_err_mc = []

    for it in range(n_it):

        flux_per = flux + rng.normal(0, fluxerr)

        if line == 1:
            try:
                popt, pcov = curve_fit(
                    lambda x, intercept, slope, flux1, mean2, flux2: model2(x, intercept, slope, 
                                                                            sigma, mean, flux1, mean2, flux2),
                    wave,
                    flux_per,
                    p0 = intial_fixed,
                    sigma = fluxerr,
                    absolute_sigma = True
                )

                intercept_mc, slope_mc, flux_mc_i1, _, _ = popt
                err = np.sqrt(np.diag(pcov))
                flux_err_i1 = err[2]

                flux_mc.append(flux_mc_i1)
                flux_err_mc.append(flux_err_i1)

            except (RuntimeError, ValueError):
                continue

        else: 
            try:
                popt, pcov = curve_fit(
                    lambda x, intercept, slope, mean1, flux1, flux2: model2(x, intercept, slope, 
                                                                            sigma, mean1, flux1, mean, flux2),
                    wave,
                    flux_per,
                    p0 = intial_fixed,
                    sigma = fluxerr,
                    absolute_sigma = True
                )

                _, _, _, _, flux_mc_i2 = popt
                err = np.sqrt(np.diag(pcov))
                flux_err_i2 = err[4]

                flux_mc.append(flux_mc_i2)
                flux_err_mc.append(flux_err_i2)

            except (RuntimeError, ValueError):
                continue

    flux_mc = np.array(flux_mc)
    flux_err_mc = np.array(flux_err_mc)

    upper_flux = 3 * np.std(flux_mc, ddof = 1)
    upper_flux_err = np.median(flux_err_mc)

    return upper_flux, upper_flux_err

def refit_flux3(wave, flux, initial, fluxerr, fixed, line):
    from scipy.optimize import curve_fit

    sigma, mean = fixed
    if line == 1:
        intercept0, slope0, _, _, flux10, mean20, flux20, mean30, flux30 = initial
        intial_fixed = [intercept0, slope0, flux10, mean20, flux20, mean30, flux30]

    elif line == 2:
        intercept0, slope0, _, mean10, flux10, _, flux20, mean30, flux30 = initial
        intial_fixed = [intercept0, slope0, mean10, flux10, flux20, mean30, flux30]

    else:
        intercept0, slope0, _, mean10, flux10, mean20, flux20, _, flux30 = initial
        intial_fixed = [intercept0, slope0, mean10, flux10, mean20, flux20, flux30]

    rng = np.random.default_rng(13)

    n_it = 100

    flux_mc = []
    flux_err_mc = []

    for it in range(n_it):

        flux_per = flux + rng.normal(0, fluxerr)

        if line == 1:
            try:
                popt, pcov = curve_fit(
                    lambda x, intercept, slope, flux1, mean2, flux2, mean3, flux3: model3(x, intercept, slope, 
                                                                            sigma, mean, flux1, mean2, flux2, mean3, flux3),
                    wave,
                    flux_per,
                    p0 = intial_fixed,
                    sigma = fluxerr,
                    absolute_sigma = True
                )

                _, _, flux_mc_i1, _, _, _, _ = popt
                err = np.sqrt(np.diag(pcov))
                flux_err_i1 = err[2]

                flux_mc.append(flux_mc_i1)
                flux_err_mc.append(flux_err_i1)

            except (RuntimeError, ValueError):
                continue

        elif line == 2: 
            try:
                popt, pcov = curve_fit(
                    lambda x, intercept, slope, mean1, flux1, flux2, mean3, flux3: model3(x, intercept, slope, 
                                                                            sigma, mean1, flux1, mean, flux2, mean3, flux3),
                    wave,
                    flux_per,
                    p0 = intial_fixed,
                    sigma = fluxerr,
                    absolute_sigma = True
                )

                _, _, _, _, flux_mc_i2, _, _ = popt
                err = np.sqrt(np.diag(pcov))
                flux_err_i2 = err[4]

                flux_mc.append(flux_mc_i2)
                flux_err_mc.append(flux_err_i2)

            except (RuntimeError, ValueError):
                continue

        else: 
            try:
                popt, pcov = curve_fit(
                    lambda x, intercept, slope, mean1, flux1, mean2, flux2, flux3: model3(x, intercept, slope, 
                                                                            sigma, mean1, flux1, mean2, flux2, mean, flux3),
                    wave,
                    flux_per,
                    p0 = intial_fixed,
                    sigma = fluxerr,
                    absolute_sigma = True
                )

                _, _, _, _, _, _, flux_mc_i3 = popt
                err = np.sqrt(np.diag(pcov))
                flux_err_i3 = err[6]

                flux_mc.append(flux_mc_i3)
                flux_err_mc.append(flux_err_i3)

            except (RuntimeError, ValueError):
                continue

    flux_mc = np.array(flux_mc)
    flux_err_mc = np.array(flux_err_mc)

    upper_flux = 3 * np.std(flux_mc, ddof = 1)
    upper_flux_err = np.median(flux_err_mc)

    return upper_flux, upper_flux_err

def fit_params(urls, survey):
    import pandas as pd
    from pathlib import Path

    plot_dir = Path("..") / "data" / survey
    plot_dir.mkdir(parents=True, exist_ok=True)

    objects = []
    fit_params = []

    for url in urls:
        print('⊹₊˚‧︵‿₊⊱·✶·⊰₊‿︵‧˚₊⊹₊˚‧︵‿₊⊱·✶·⊰₊‿︵‧˚₊⊹')
        print('')
        resampled = resampling_spectra(url) #For example: "aurora-gdn02-v4_g235m-f170lp_1914_931951.spec.fits"

        averaged = weighted_average(resampled)

        plot_spectrum(averaged, resampled)

        root = averaged.meta["root"]
        srcid = averaged.meta["srcid"]
        identification = root +'_'+ srcid

        fits = {}
        fluxes = {}
        fluxerrs = {}

        for line in lines:
            optimal, opterrs = line_cont_fit(averaged, resampled, line)

            if len(optimal.keys()) == 0 or len(opterrs.keys()) == 0:
                print(f"Skipping {line['element']}: fit unavailable.")
                continue

            n_lines = optimal["lines"]
            intercept = optimal.get("intercept")
            intercept_err = opterrs.get("intercept")
            slope = optimal.get("slope")
            slope_err = opterrs.get("slope")
            sigma = optimal.get("sigma")
            sigma_err = opterrs.get("sigma")

            for i in range(n_lines):
                idx = i + 1
                elem_name = optimal["element"][i]

                # 1. Keep your existing wide flux table logic
                fluxes[elem_name] = optimal[f"flux{idx}"]
                fluxerrs[elem_name] = opterrs[f"flux{idx}"]

                # 2. Extract all fitted parameters for this Gaussian component
                fits[(elem_name, "intercept")] = intercept
                fits[(elem_name, "intercept_err")] = intercept_err
                fits[(elem_name, "slope")] = slope
                fits[(elem_name, "slope_err")] = slope_err
                fits[(elem_name, "sigma")] = sigma
                fits[(elem_name, "sigma_err")] = sigma_err

                for param in ["flux", "mean"]:
                    fits[(elem_name, param)] = optimal.get(
                        f"{param}{idx}"
                    )
                    fits[(elem_name, f"{param}_err")] = opterrs.get(
                        f"{param}{idx}"
                    )


        fluxes = pd.DataFrame.from_dict(fluxes, orient = 'index', columns = ['flux'])
        fluxerrs = pd.DataFrame.from_dict(fluxerrs, orient = 'index', columns = ['flux_err'])
        object_df = pd.concat([fluxes, fluxerrs], axis = 1).stack().to_frame().T
        object_df.index = [identification]
        objects.append(object_df)

        if fits:
            param_df = pd.DataFrame([fits], index=[identification])
            param_df.columns = pd.MultiIndex.from_tuples(
                param_df.columns, names=["element", "parameter"]
            )
            fit_params.append(param_df)

        print('')

    final_fluxes = pd.concat(objects)
    final_fits = pd.concat(fit_params)
    final_fluxes.index.name = 'object'
    final_fits.index.name = 'object'
    final_fluxes.to_parquet(plot_dir / 'Fluxes.parquet')
    final_fits.to_parquet(plot_dir / 'Fit_params.parquet')
    return final_fluxes
    
def abundance(flux_df):
    import pyneb as pn
    import numpy as np
    import pandas as pd

    from pathlib import Path


    z_dir = Path("..") / "abundances" 
    z_dir.mkdir(parents=True, exist_ok=True)

    O3 = pn.Atom("O", 3)
    O2 = pn.Atom("O", 2)
    S2 = pn.Atom("S", 2)

    all_objs = {}

    s2_tem_ex = 1e4 # Chosen by hand (how to use effective collision strengths?)

    np.random.seed(1312)
    nrand = 1000

    for obj, row in flux_df.iterrows():

        def has_lines(*elems):
            return all(
            (elem, "flux") in row.index
            and (elem, "flux_err") in row.index
            and pd.notna(row[(elem, "flux")])
            and pd.notna(row[(elem, "flux_err")])
            for elem in elems
        )

        def get_stack(elem):
            # Remove '* 1e20' if flux and flux_err are already in the exact same units
            return (
                row[(elem, "flux")]
                + np.random.normal(0, 1, nrand) * row[(elem, "flux_err")]
            )        
        
        file = open(z_dir / f"{obj}_metallicity.txt", 'w')
        obj_dict = {}

        print('⊹₊˚‧︵‿₊⊱·✶·⊰₊‿︵‧˚₊⊹₊˚‧︵‿₊⊱·✶·⊰₊‿︵‧˚₊⊹')
        print('')
        print(obj)
        print('')

        # print("SII:", has_lines('[SII]6717', '[SII]6731'))
        # print("OII nebular:", has_lines('[OII]3727'))
        # print("OII auroral:", has_lines('[OII]7322', '[OII]7332'))
        # print("OIII:", has_lines('[OIII]4364', '[OIII]5007'))
        # print("Hbeta:", has_lines('Hbeta'))

        # print(row.index)

        if has_lines('[SII]6717', '[SII]6731'):

            sii6717_stack = get_stack('[SII]6717')
            sii6731_stack = get_stack('[SII]6731')

            s2_den = S2.getTemDen(sii6717_stack/sii6731_stack, tem = s2_tem_ex, wave1=6716,wave2=6730)

            s2den_16, s2den_50, s2den_84 = np.nanpercentile(s2_den,(15.8,50,84.2))

            obj_dict['s2_den'] = [s2den_16, s2den_50, s2den_84]

            file.write('Density estimates:\n')
            file.write(f'{s2den_16} {s2den_50} {s2den_84}\n')

        else: s2_den = 100

        if has_lines('[OII]7322', '[OII]7332'):
            oii7322_stack = get_stack('[OII]7322')
            oii7332_stack = get_stack('[OII]7332')    

        if has_lines('[OII]3727'):
            oii3727_stack = get_stack('[OII]3727')

            if has_lines('[OII]7322', '[OII]7332'):
                o2_temp = O2.getTemDen((oii7322_stack + oii7332_stack)/oii3727_stack, 
                                    den = s2_den, to_eval = '(L(7320) + L(7330))/(L(3726) + L(3729))')

                o2temp_16,o2temp_50,o2temp_84 = np.nanpercentile(o2_temp,(15.8,50,84.2))
                
                obj_dict['o2_tem'] = [o2temp_16, o2temp_50, o2temp_84]

                file.write('Te O+ estimates:\n')
                file.write(f'{o2temp_16} {o2temp_50} {o2temp_84}\n')
        else: 
            file.write('O+ nebular line measurements unavailable\n')


        if (has_lines('[OIII]4364') and has_lines('[OIII]5007')):
            oiii4364_stack = get_stack('[OIII]4364')
            oiii5007_stack = get_stack('[OIII]5007')

            o3_temp = O3.getTemDen(oiii4364_stack/oiii5007_stack, 
                                den = s2_den, wave1=4364,wave2=5007)

            o3temp_16, o3temp_50, o3temp_84 = np.nanpercentile(o3_temp,(15.8,50,84.2))

            obj_dict['o3_tem'] = [o3temp_16, o3temp_50, o3temp_84]

            file.write('Te O2+ estimates:\n')
            file.write(f'{o3temp_16} {o3temp_50} {o3temp_84} \n')

            o2_temp_rel = 0.7 * o3_temp + 3000
            
            o2temprel_16,o2temprel_50,o2temprel_84 = np.nanpercentile(o2_temp_rel,(15.8,50,84.2))
        
            obj_dict['o2_rel_tem'] = [o2temprel_16, o2temprel_50, o2temprel_84]
        
            file.write('Te O+ empirical estimates:\n')
            file.write(f'{o2temprel_16} {o2temprel_50} {o2temprel_84} \n')

        # else: 
        #     file.write('O2+ line measurements unavailable\n')
        #     file.close()
        #     return object_dict

        
        if has_lines('Hbeta', '[OIII]5007') and 'o3_tem' in obj_dict:
            hbeta_stack = get_stack('Hbeta')

            o3_abun = O3.getIonAbundance(oiii5007_stack/hbeta_stack*100, tem=o3_temp, den=s2_den,
                                        wave=5007)

            if has_lines('[OII]3727'):
                o2_abun = O2.getIonAbundance(oii3727_stack/hbeta_stack*100, 
                                            tem=o2_temp_rel, 
                                            den=s2_den, 
                                            wave = 3727
                                            )
                
            elif has_lines('[OII]7322', '[OII]7332'):
                o2_abun = O2.getIonAbundance((oii7322_stack + oii7332_stack) / hbeta_stack * 100, 
                                            tem=o2_temp_rel, 
                                            den=s2_den, 
                                            to_eval='L(7319)+L(7320)+L(7329)+L(7331)'
                                            )

            else:
                o2_abun = None
                file.write("O+ abundance unavailable: no usable [OII] lines\n")


            obj_dict['o3_abun'] = o3_abun
            if o2_abun is not None:
                obj_dict['o2_abun'] = o2_abun

                o_abun = o3_abun+o2_abun
                o_abun_16, o_abun_50, o_abun_84 = np.nanpercentile(o_abun,(15.8,50,84.2))

                obj_dict['o_abun'] = [o_abun_16, o_abun_50, o_abun_84]

                file.write(f'Stack has 12+log(O/H) = {12+np.log10(o_abun_50):.2f} ({12+np.log10(o_abun_16):.2f}-{12+np.log10(o_abun_84):.2f}) \n')

                obj_dict['12+log(O/H)'] = [12+np.log10(o_abun_16), 12+np.log10(o_abun_50), 12+np.log10(o_abun_84)]

            file.close()

        else: 
            file.write('H beta measurements not available\n')
            file.close() 

        all_objs[obj] = obj_dict

    abun_df = pd.DataFrame.from_dict(
        all_objs,
        orient='index'
    )

    abun_df.index.name = 'object'

    abun_df.to_parquet(z_dir / 'Abundances.parquet')

    return abun_df
    

    