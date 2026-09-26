def line_cont_fit(spectrum, line_info):
    from astropy import units as u
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from astropy.nddata import StdDevUncertainty

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
                

    z = spectrum.meta['z']

    print(line_info['element'])

    ###############################################################################################

    ## Centering + masking (maybe needs change)

    if line_info['element'][0] == '1':
        center = line_info['rest_AA'] * (1+z)
    else:
        center = line_info['rest_AA'][0] * (1+z)
    window = 100 * u.AA  # Provisional window, subject to change

    mask = ((spectrum.spectral_axis >= center - window) & 
            (spectrum.spectral_axis <= center + window))

    wave_masked = spectrum.spectral_axis[mask]
    flux_masked = spectrum.flux[mask]
    # flux_err_masked = spectrum.uncertainty.represent_as(StdDevUncertainty).array[mask]
    
    flux_masked = flux_masked.to(u.erg / u.cm**2) / (wave_masked**2) * (3*1e18 * (u.AA / u.s))  
    flux_err_masked = spectrum.uncertainty.quantity[mask].to(u.erg / u.cm**2) / (wave_masked**2) * (3*1e18 * (u.AA / u.s))  
    flux_err_masked = StdDevUncertainty(flux_err_masked).array

    ###############################################################################################

    ## Initial values for curvefit (1 gaussian, 2 gaussians or 3 gaussians)

    if line_info['element'][0] == '1':
        initial = [np.median(flux_masked.value), # The median of the flux as a guess for the intercept
                    0,                            # The slope is practically null for the continuum
                    line_info['sigma'],                          # Sigma, assuming most of the lines aren't broad components
                    line_info['rest_AA'].value * (1 + z),              # Rest in AA * (redshift correction since the wave array is observed)
                    line_info['sigma'] * np.sqrt(2*np.pi) * line_info['amplitude']]       # Then the expected flux, not amplitude

    if line_info['element'][0] == '2':
        initial = [np.median(flux_masked.value), 
                    0,                           
                    line_info['sigma'],
                    line_info['rest_AA'][0].value * (1 + z),
                    line_info['sigma'] * np.sqrt(2*np.pi) * line_info['amplitude'][0],
                    line_info['rest_AA'][1].value * (1 + z),
                    line_info['sigma'] * np.sqrt(2*np.pi) * line_info['amplitude'][1]]

    if line_info['element'][0] == '3':
        initial = [np.median(flux_masked.value), 
                    0,                           
                    line_info['sigma'],
                    line_info['rest_AA'][0].value * (1 + z),
                    line_info['sigma'] * np.sqrt(2*np.pi) * line_info['amplitude'][0],
                    line_info['rest_AA'][1].value * (1 + z),
                    line_info['sigma'] * np.sqrt(2*np.pi) * line_info['amplitude'][1], 
                    line_info['rest_AA'][2].value * (1 + z),
                    line_info['sigma'] * np.sqrt(2*np.pi) * line_info['amplitude'][2]]

    ###############################################################################################

    ## Curvefit with the different models (1, 2, or 3 gaussians)
    
    wave_model = np.linspace(
        wave_masked.value.min(),
        wave_masked.value.max(),
        5000
    )

    if line_info['element'][0] == '1':
        popt, pcov = curve_fit(model1, wave_masked.value, flux_masked.value, p0 = initial, 
                        sigma = flux_err_masked, absolute_sigma = True)

        flux_model = model1(wave_model, *popt)

    if line_info['element'][0] == '2':
        popt, pcov = curve_fit(model2, wave_masked.value, flux_masked.value, p0 = initial, 
                        sigma = flux_err_masked, absolute_sigma = True)

        flux_model = model2(wave_model, *popt)

    if line_info['element'][0] == '3':
        popt, pcov = curve_fit(model3, wave_masked.value, flux_masked.value, p0 = initial, 
                        sigma = flux_err_masked, absolute_sigma = True)

        flux_model = model3(wave_model, *popt)

    ###############################################################################################

    ## Plotting!

    f, ax = plt.subplots(figsize = (20, 3))

    ax.plot(wave_masked, flux_masked, color = 'midnightblue')

    ax.plot(wave_model, flux_model, color = 'r', alpha = 0.5)
    ax.minorticks_on()
  
    # ax.set_xlim(0.48, 0.51)
    # ax.set_ylim(-1, 20)
    ax.set_title(f'Isolated {line_info['element'][1:]} fitting')
    ax.grid(alpha = 0.5)
    plt.show()

    ###############################################################################################

    ## Optimal values and errors for each amount of gaussian case
    ## important to consider which are the 1st, 2nd and 3rd elements in a list
    
    if line_info['element'][0] == '1':
        intercept, slope, sigma, mean, flux = popt

        optimal_dict = {
            'element': line_info['element'][1:],
            'lines': line_info['element'][0],
            'intercept': intercept, 
            'slope': slope, 
            'sigma': sigma, 
            'mean1': mean,
            'flux1': flux,
        }

        model_errs = np.sqrt(np.diag(pcov))
        intercept_err, slope_err, sigma_err, mean_err, flux_err = model_errs

        opterrs_dict = {
            'intercept': intercept_err, 
            'slope': slope_err, 
            'sigma': sigma_err, 
            'mean': mean_err,
            'flux': flux_err,
        }

    if line_info['element'][0] == '2':
        intercept, slope, sigma, mean1, flux1, mean2, flux2 = popt

        optimal_dict = {
            'element': line_info['element'][1:],
            'lines': line_info['element'][0],
            'intercept': intercept, 
            'slope': slope, 
            'sigma': sigma, 
            'mean1': mean1,
            'flux1': flux1,
            'mean2': mean2,
            'flux2': flux2,
        }

        model_errs = np.sqrt(np.diag(pcov))
        intercept_err, slope_err, sigma_err, mean_err1, flux_err1, mean_err2, flux_err2 = model_errs

        opterrs_dict = {
            'intercept': intercept_err, 
            'slope': slope_err, 
            'sigma': sigma_err, 
            'mean1': mean_err1,
            'flux1': flux_err1,
            'mean2': mean_err2,
            'flux2': flux_err2,
        }
    if line_info['element'][0] == '3':
        intercept, slope, sigma, mean1, flux1, mean2, flux2, mean3, flux3 = popt
    
        optimal_dict = {
            'element': line_info['element'][1:],
            'lines': line_info['element'][0],
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

        model_errs = np.sqrt(np.diag(pcov))
        intercept_err, slope_err, sigma_err, mean_err1, flux_err1, mean_err2, flux_err2, mean_err3, flux_err3 = model_errs

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

    return [optimal_dict, opterrs_dict]