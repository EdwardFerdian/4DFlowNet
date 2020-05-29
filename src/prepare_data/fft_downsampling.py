import time
import math
import numpy as np


def rectangular_crop3d(f, crop_ratio):
    half_x = f.shape[0] // 2
    half_y = f.shape[1] // 2
    half_z = f.shape[2] // 2
    
    # print('half', half_x, half_y, half_z)

    x_crop = int(half_x * crop_ratio)
    y_crop = int(half_y * crop_ratio)
    z_crop = int(half_z * crop_ratio)

    # shift it to make it easier to crop, otherwise we need to concat half left and half right
    new_kspace = np.fft.fftshift(f)
    new_kspace = new_kspace[half_x-x_crop:half_x+x_crop, half_y-y_crop:half_y+y_crop, half_z-z_crop : half_z+z_crop]
    # shift it back to original freq domain
    new_kspace = np.fft.fftshift(new_kspace)
     
    return new_kspace


def add_complex_signal_noise(imgfft, targetSNRdb):
    """
        Add gaussian noise to real and imaginary signal
        The sigma is assumed to be the same (Gudbjartsson et al. 1995)

        SNRdb = 10 log SNR
        SNRdb / 10 = log SNR
        SNR = 10 ^ (SNRdb/10)
        
        Pn = Pn / SNR
        Pn = variance
        
        Relation of std and Pn is taken from Matlab Communication Toolbox, awgn.m

        For complex signals, we can use the equation above.
        If we do it in real and imaginary signal, half the variance is in real and other half in in imaginary.
        
        https://www.researchgate.net/post/How_can_I_add_complex_white_Gaussian_to_the_signal_with_given_signal_to_noise_ratio
        "The square of the signal magnitude is proportional to power or energy of the signal.
        SNR is the ratio of this power to the variance of the noise (assuming zero-mean additive WGN).
        Half the variance is in the I channel, and half is in the Q channel.  "

    """    
    add_complex_noise =True
    # adding noise on the real and complex image
    # print("--------------Adding Gauss noise to COMPLEX signal----------------")

    # Deconstruct the complex numbers into real and imaginary
    mag_signal = np.abs(imgfft)
    
    signal_power = np.mean((mag_signal) ** 2)

    logSNR = targetSNRdb / 10
    snr = 10 ** logSNR

    noise_power = signal_power / snr

    if add_complex_noise:
        sigma  = np.sqrt(noise_power)
        # print('Target SNR ', targetSNRdb, "db, sigma(complex)", sigma)

        # add the noise to the complex signal directly
        gauss_noise = np.random.normal(0, sigma, imgfft.shape)
        imgfft += gauss_noise
    else:
        # Add the noise to real and imaginary separately
        sigma  = np.sqrt(noise_power/2)
        # print('Target SNR ', targetSNRdb, "db, sigma (real/imj)", sigma)
        
        real_signal = np.real(imgfft)
        imj_signal = np.imag(imgfft)
        
        real_noise = np.random.normal(0, sigma, real_signal.shape)
        imj_noise  = np.random.normal(0, sigma, imj_signal.shape)
        
        # add the noise to both components
        real_signal = real_signal + real_noise
        imj_signal  = imj_signal + imj_noise
        
        # reconstruct the image back to complex numbers
        imgfft = real_signal + 1j * imj_signal

    return imgfft

def downsample_complex_img(complex_img, crop_ratio, targetSNRdb):
    imgfft = np.fft.fftn(complex_img)

    imgfft = rectangular_crop3d(imgfft, crop_ratio)
    
    shifted_mag  = 20*np.log(np.fft.fftshift(np.abs(imgfft)))

    # add noise on freq domain
    imgfft = add_complex_signal_noise(imgfft, targetSNRdb)

    # inverse fft to image domain
    new_complex_img = np.fft.ifftn(imgfft)

    return new_complex_img, shifted_mag


def rescale_magnitude_on_ratio(new_mag, old_mag):
    old_mag_flat = np.reshape(old_mag, [-1])
    new_mag_flat = np.reshape(new_mag, [-1])

    rescale_ratio = new_mag_flat.shape[0] / old_mag_flat.shape[0]

    return new_mag * rescale_ratio
    
def downsample_phase_img(velocity_img, mag_image, venc, crop_ratio, targetSNRdb):
    # convert to phase
    phase_image = velocity_img / venc * math.pi

    complex_img = np.multiply(mag_image, np.exp(1j*phase_image))
    
    # -----------------------------------------------------------
    new_complex_img, shifted_freqmag = downsample_complex_img(complex_img, crop_ratio, targetSNRdb)
    # -----------------------------------------------------------

    # Get the MAGnitude and rescale
    new_mag = np.abs(new_complex_img)
    new_mag = rescale_magnitude_on_ratio(new_mag, mag_image)

    # Get the PHASE
    new_phase = np.angle(new_complex_img)

    # Get the velocity image
    new_velocity_img = new_phase / math.pi * venc

    return new_velocity_img, new_mag