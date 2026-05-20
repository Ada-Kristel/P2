import ismrmrd
import numpy as np
import matplotlib.pyplot as plt

"""
Den her fil printer rekonstruktionerne med ekstra støj.
Støj (standardafvigelse) kan ændres på linje 88.
For at skifte mellem zerofill og least squares, skal udkommenteringen byttes om ved linje 267-268.
"""

# Finder filen
path = r"C:/Data/2dknee.h5"

# Tildeler filen som en array
dset = ismrmrd.Dataset(path, "dataset")

# Læser information om encoding fra xml filen
header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
enc = header.encoding[0]

kx = enc.encodedSpace.matrixSize.x # Definerer størrelsen af matricen i kx retning - 352
ky_size = enc.encodedSpace.matrixSize.y # Definerer størrelsen af matricen i ky retning - 202

# Vælger 1 coil fremfor alle 16, så vi kan få en nx x ny matrix
# Indekset bestemmer hvilken af de 16 coils (0-15) vi vælger
coil_number = 0

# Vi kigger kun på en slice
# Indekset her bestemmer hvilken af de 28 slices (0-27) vi vælger
slice_number = 10

# Generer en tom 2d matrix af størrelsen ny x nx.
# np.complex64 sørger for at hver indgang i matricen ligner og agerer som et komplekst tal
kspace = np.zeros((ky_size,kx),dtype=np.complex64)

def fillkspace():
    """
    Der bliver kørt igennem alle acquisitions (samples)
    og fylder række for række for den valgte slice og den valgte coil
    Hvis den slice vi kigger på ikke er vores valgte slice, går den videre til næste iteration
    """
    for i in range(dset.number_of_acquisitions()):
        acq = dset.read_acquisition(i)  # Vælger hvilken acquisition vi kigger på

        if acq.idx.slice != slice_number:
            continue

        # kspace_encode_step_1 fortæller os i hvilket row den enkelte acquisition hører til.
        row = acq.idx.kspace_encode_step_1
        # For at filtrere noget støj (eller kalibrering) væk sørger vi for kun at kigge indenfor vores matrixsize
        # hvis vi får en row værdi som er negativ eller større end de 202, ser vi bort fra den
        if row < 0 or row >= ky_size:
            continue

        """
        data svarer til acquisition data i vores acq array, som indeholder alt dataet for hver sample
        # hver acquisition har coil number og tilsvarende samples.
        # vi sætter coil number til en specifik værdi og får dermed kun vores samples på linjen.
        """
        line = acq.data[coil_number, :]

        """
        n = min(n,kx)
        Vælger den mindste værdi af enten kx eller line.shape[0].
        nx er bredden af vores k-space som defineret tidligere.
        line.shape[0] kigger på dimensionen af det line array, hvilket er bredden altså antal indgange i arrayen
        Det gør vi for at begrænse vores opfyldning af k-space til den størrelse, 
        som vi har defineret den til at være. (Ser bort fra eventuel støj osv.)
        """
        n = min(kx, line.shape[0])

        """
        Den line af kx værdier, som vi har bestemt bliver tildelt hvert row, da row bliver indekseret.
        Bredden af matricen kspace og bredden af vektoren line bliver begge begrænset af den samme værdi n.
        Så en line svarer til en linje i kspace.
        """
        kspace[row, :n] = line[:n]

    return kspace

def transform(kspace):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))
image = transform(fillkspace())

# tilføjer støj til kspace
kspace = fillkspace()
np.random.seed(37)
noise = 2E05
kspace = kspace + np.random.normal(0, noise, kspace.shape) + 1j * np.random.normal(0, noise, kspace.shape)

def samplingmask(ky_size, center_fraction = 0.30):
    """
    :param ky_size: antal rækker i datasættet
    :param center_fraction: bestemmer hvor stor en del af rækkerne, som centrum udgør
    :return: en maske til at undersample datasættet
    """
    mask = np.zeros(ky_size, dtype=bool)

    # Beregner start og slut indeks for centrum af k-space
    center_start = ky_size // 2 - int(ky_size * center_fraction) // 2
    center_end = ky_size // 2 + int(ky_size * center_fraction) // 2
    mask[center_start:center_end] = True

    n_center = mask.sum()
    print(f'Rækker beholdt: {n_center: .0f}')  # printer hvor mange rækker vi beholder ud af de 202

    mask[mask] = True

    return mask, n_center
mask, n_center = samplingmask(ky_size)

def undersampling(kspace):
    mask, _ = samplingmask(ky_size)
    kspace_undersampled = kspace.copy()  # undersampler en kopi af kspace
    kspace_undersampled[~mask, :] = 0
    return kspace_undersampled

kspace_undersampled = undersampling(kspace)
image_undersampled = transform(kspace_undersampled)

def Leastsquares(kspace_undersampled, mask_1d, delta):
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
    return transform(kspace_weighted), kspace_weighted, delta

# for reference
# zerofill RMSE ved 1E+04 støj er 0.0187, som er meget tæt på RMSE for ingen støj (0.0186)
# zerofill RMSE ved 1E+05 støj er 0.0600
# zerofill RMSE ved 2E+05 støj er 0.2348
# zerofill RMSE ved 5E+05 støj er 1.7100
# zerofill RMSE ved 1E+06 støj er 7.6633

# delta ændrer på betydningen af regulariseringstermet
#delta = 0.00001
#delta = 0.25
#delta = 0.0629 # den bedste delta for en støj på 1E+05 RMSE = 0.0562
delta = 0.2770 # den bedste delta for en støj på 2E+05 RMSE = 0.1704
#delta = 1.5301 # den bedste delta for en støj på 5E+05 RMSE = 0.4697
#delta = 4.3016 # den bedste delta for en støj på 1E+06
image_L2, kspace_L2, _ = Leastsquares(kspace_undersampled,mask,delta)

def RelativeMeanSquareError(image_ref, image_recon):
    """
    :param image_ref: Reference image ("perfekte" billede)
    :param image_recon: Reconstructed image
    :return: Returnerer de relative fejl mellem det "perfekte" billede og rekonstruktionen
    """
    ref = np.abs(image_ref)
    recon = np.abs(image_recon)

    return np.sum(np.square(ref - recon)) / np.sum(np.square(ref))

# min og max for grayscale (kspace)
vmin_k = 5.365233
vmax_k = 17.216454

# min og max for grayscale (image)
vmin_i = 0.9819881
vmax_i = 3827.4158


# indsamler info til tabellerne
n_kept = int(mask.sum())
n_percentile = float(n_kept/ky_size*100)

def zerofillprint(kspace_undersampled,image_undersampled):
    rmse = RelativeMeanSquareError(image, image_undersampled)

    fig = plt.figure(figsize=(14, 8))

    # laver et gridspec: øverste række for billeder, nederste række til tabellen
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], hspace=-0.4, wspace=0.1)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :])  # spans both columns

    # plotter kspace
    ax0.imshow(np.log(np.abs(kspace_undersampled) + 1E-09), cmap='gray', vmin=vmin_k, vmax=vmax_k)
    ax0.set_title("Undersampled k-space")

    # plotter det rekonstruerede billede - skift mellem least squares og zerofill
    ax1.imshow(np.abs(image_undersampled), cmap='gray', vmin=vmin_i, vmax=vmax_i)
    ax1.set_title("Zerofill reconstructed image")

    ax0.axis('off')
    ax1.axis('off')

    # bygger tabellen
    table_data = [
        ["Rows kept", f"{n_kept} / {ky_size}"],
        ["Percentage kept", f"{n_percentile:.2f}%"],
        ["RMSE", f"{rmse:.4f}"],
    ]

    ax_table.axis('off')
    table = ax_table.table(
        cellText=table_data,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(0.5, 1.5)  # ændrer på størrelsen af tabellen

    plt.show()

def leastsquaresquaresprint(kspace_undersampled, image_l2):
    rmse = RelativeMeanSquareError(image, image_l2)

    fig = plt.figure(figsize=(14, 8))

    # laver et gridspec: øverste række for billeder, nederste række til tabellen
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], hspace=-0.4, wspace=0.1)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :])  # spans both columns

    # plotter kspace
    ax0.imshow(np.log(np.abs(kspace_undersampled) + 1E-09), cmap='gray', vmin=vmin_k, vmax=vmax_k)
    ax0.set_title("Undersampled k-space")

    # plotter det rekonstruerede billede
    ax1.imshow(np.abs(image_L2), cmap='gray',vmin = vmin_i, vmax = vmax_i)
    ax1.set_title("L2 reconstructed image")

    ax0.axis('off')
    ax1.axis('off')

    # bygger tabellen
    table_data = [
        ["Rows kept", f"{n_kept} / {ky_size}"],
        ["Percentage kept", f"{n_percentile:.2f}%"],
        ["RMSE", f"{rmse:.4f}"],
        ["$\delta$", f"{delta}"],
    ]

    ax_table.axis('off')
    table = ax_table.table(
        cellText=table_data,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(0.5, 1.5)  # ændrer på størrelsen af tabellen

    plt.show()

zerofillprint(kspace_undersampled, image_undersampled)
#leastsquaresquaresprint(kspace_undersampled, image_L2)
