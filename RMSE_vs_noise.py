import ismrmrd
import numpy as np
import matplotlib.pyplot as plt
"""
Filen printer RMSE mod forskellige standardafvigelseer for både zerofill og least squares.
Dertil findes punktet, hvor de to grafer krydser.
"""

path = r"C:/Data/2dknee.h5"
dset = ismrmrd.Dataset(path, "dataset")
header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
enc = header.encoding[0]

kx      = enc.encodedSpace.matrixSize.x   # 352
ky_size = enc.encodedSpace.matrixSize.y   # 202

coil_number  = 0
slice_number = 10


def fillkspace():
    """
    Der bliver kørt igennem alle acquisitions (samples)
    og fylder række for række for den valgte slice og den valgte coil
    Hvis den slice vi kigger på ikke er vores valgte slice, går den videre til næste iteration
    """
    kspace = np.zeros((ky_size, kx), dtype=np.complex64)
    for i in range(dset.number_of_acquisitions()):
        acq = dset.read_acquisition(i)
        if acq.idx.slice != slice_number:
            continue
        row = acq.idx.kspace_encode_step_1
        if row < 0 or row >= ky_size:
            continue
        line = acq.data[coil_number, :]
        n = min(kx, line.shape[0])
        kspace[row, :n] = line[:n]
    return kspace

def transform(kspace):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))

def samplingmask(ky_size, center_fraction=0.30):
    """
    param ky_size: antal rækker i datasættet
    param center_fraction: bestemmer hvor stor en del af rækkerne, som centrum udgør
    return: en maske til at undersample datasættet
    """
    mask = np.zeros(ky_size, dtype=bool)
    cs = ky_size // 2 - int(ky_size * center_fraction) // 2
    ce = ky_size // 2 + int(ky_size * center_fraction) // 2
    mask[cs:ce] = True
    return mask

def Leastsquares(kspace_undersampled, mask_1d, delta=0.01):
    """
    Regulariseret mindste kvadraters metode med l2
    Funktionen løser: min ||MFx - b||^2 + delta*||x||^2

    Parameters:
        kspace_undersampled : undersampled k-space
        mask_1d             : 1d maske, som tage masken pr. række
        delta               : regulariseringsstyrken

    Returns:
        Rekonstrueret billede
    """
    ky, kx = kspace_undersampled.shape
    mask_2d = np.tile(mask_1d[:, np.newaxis].astype(np.float32), (1, kx))

    weights = 1.0 / (mask_2d + delta)

    kspace_weighted = weights * kspace_undersampled
    return transform(kspace_weighted), kspace_weighted

def RelativeMeanSquareError(image_ref, image_recon):
    """
    param image_ref: Reference image ("perfekte" billede)
    param image_recon: Reconstructed image
    return: Returnerer de relative fejl mellem det "perfekte" billede og rekonstruktionen
    """
    ref = np.abs(image_ref)
    recon = np.abs(image_recon)

    return np.sum(np.square(ref - recon)) / np.sum(np.square(ref))


print("Loading k-space …")
kspace_clean = fillkspace()
image_perfect = transform(kspace_clean)

mask = samplingmask(ky_size)


kspace_clean_us = kspace_clean.copy()
kspace_clean_us[~mask, :] = 0


noise_stds   = np.logspace(np.log10(1e4), np.log10(1e6), 30)   # 30 støjniveauer
delta_grid   = np.logspace(-4, 2, 60)                     # 60 delta-værdi, som bliver undersøgt

rng = np.random.default_rng(37)

rmse_zerofill = []
rmse_l2_best  = []
best_deltas   = []

print(f"\nRunning simulation over {len(noise_stds)} noise levels …\n")

for idx, std in enumerate(noise_stds):
    # Add noise to k-space
    noise_re = rng.normal(0, std, kspace_clean.shape)
    noise_im = rng.normal(0, std, kspace_clean.shape)
    kspace_noisy = kspace_clean + noise_re + 1j * noise_im

    # Under-sample
    kspace_noisy_us = kspace_noisy.copy()
    kspace_noisy_us[~mask, :] = 0

    # Zerofill RMSE
    img_zf = transform(kspace_noisy_us)
    rmse_zf = RelativeMeanSquareError(image_perfect, img_zf)
    rmse_zerofill.append(rmse_zf)

    # søger efter bedste delta
    best_err   = np.inf
    best_delta = delta_grid[0]
    for d in delta_grid:
        img_l2, _ = Leastsquares(kspace_noisy_us, mask, d)
        err    = RelativeMeanSquareError(image_perfect, img_l2)
        if err < best_err:
            best_err   = err
            best_delta = d

    rmse_l2_best.append(best_err)
    best_deltas.append(best_delta)

    print(f"  [{idx+1:2d}/{len(noise_stds)}]  std={std:.2e}  "
          f"best δ={best_delta:.4f}  RMSE_L2={best_err:.4f}  RMSE_ZF={rmse_zf:.4f}")

#Plotter figuren
fig, ax = plt.subplots(figsize=(8, 5))

#Zerofill graf
ax.plot(noise_stds, rmse_zerofill,
        color='steelblue', linewidth=2, label='Zerofill')
#RMSE graf
ax.plot(noise_stds, rmse_l2_best,
        color='hotpink', linewidth=2, label='L2 (best δ)')

ax.set_xscale('log')
ax.set_xlabel('Noise (SD)', fontsize=13)
ax.set_ylabel('RMSE', fontsize=13)
ax.set_title('RMSE vs Noise Level', fontsize=14)

# finder, hvor zerofill og least squares krydser
diff = np.array(rmse_zerofill) - np.array(rmse_l2_best)
sign_changes = np.where(np.diff(np.sign(diff)))[0]
if len(sign_changes) > 0:
    i = sign_changes[0]
    # Linear interpolation in log-space to find crossing x
    x0, x1 = np.log10(noise_stds[i]), np.log10(noise_stds[i + 1])
    d0, d1 = diff[i], diff[i + 1]
    x_cross = 10 ** (x0 - d0 * (x1 - x0) / (d1 - d0))
    ax.axvline(x=x_cross, color='gray', linestyle='--', linewidth=1.5,
               label=f'Intersection (SD ≈ {x_cross:.2e})')
ax.grid(True)
ax.legend(fontsize=11)
plt.tight_layout()
plt.show()
