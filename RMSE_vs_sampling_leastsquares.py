import ismrmrd
import numpy as np
import matplotlib.pyplot as plt

"""
Koden plotter relative mean square error mod samplingsprocent for least squares rekonstruktion
med 3 forskellige delta-værdier.
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
slice_number = 10 # Vælger 10 da det er et af de slices hvor man kan se mest på

# Generer en tom 2d matrix af størrelsen ny x nx.
# np.complex64 sørger for at hver indgang i matricen ligner og agerer som et komplekst tal
kspace = np.zeros((ky_size,kx),dtype=np.complex64)

def fillkspace():
    """
    Der bliver kørt igennem alle acquisitions (samples)
    og fylder række for række for den valgte slice og den valgte coil
    Hvis den slice vi kigger på ikke er vores valgte slice går den videre til næste iteration
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
        hver acquisition har coil number og tilsvarende samples.
        vi sætter coil number til en specifik værdi og får dermed kun vores samples på linjen.
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

def samplingmask(ky_size, center_fraction=0.20):
    """
    param ky_size: antal rækker i datasættet
    param center_fraction: bestemmer hvor stor en del af rækkerne, som centrum udgør
    return: en maske til at undersample datasættet
    """
    mask = np.zeros(ky_size, dtype=bool) # ændr den fra zeros til ones, hvis vi vil sample ude fra og ind mod centrum

    # Beregner start og slut indeks for centrum af k-space
    center_start = ky_size // 2 - int(ky_size * center_fraction) // 2
    center_end = ky_size // 2 + int(ky_size * center_fraction) // 2
    mask[center_start:center_end] = True # ændr fra True til False, hvis vi vil sample ude fra og ind mod centrum

    # Beregner hvor mange linjer vi vil sample udenfor centrum
    n_center = mask.sum()
    print(f'Rækker beholdt: {n_center: .0f}') # printer hvor mange rækker vi beholder ud af de 202

    return mask, n_center

def undersampling(kspace):
    mask = samplingmask(ky_size)
    kspace_undersampled = kspace.copy()  # undersampler en kopi af kspace
    kspace_undersampled[~mask, :] = 0
    return kspace_undersampled

def MeanSquareError(image_ref, image_recon):
    """
    param image_ref: Reference image ("perfekte" billede)
    param image_recon: Reconstructed image
    return: Returnerer den absolutte fejl mellem det "perfekte" billede og rekonstruktionen
    men med en skalering på den højeste reference værdi, da det er meget høje værdier
    """
    ref = np.abs(image_ref)
    recon = np.abs(image_recon)

    return np.mean(np.square(ref - recon))

def RelativeMeanSquareError(image_ref, image_recon):
    """
    param image_ref: Reference image ("perfekte" billede)
    param image_recon: Reconstructed image
    return: Returnerer de relative fejl mellem det "perfekte" billede og rekonstruktionen
    men med en skalering, da det er meget høje værdier
    """
    ref = np.abs(image_ref)
    recon = np.abs(image_recon)

    return np.sum(np.square(ref - recon)) / np.sum(np.square(ref))

# Transformerer det fulde kspace til et billede
image = transform(fillkspace())

def Leastsquares(kspace_undersampled, mask_1d, delta):
    """
    Regularized least squares reconstruction with L2 penalty.
    Solves: min ||MFx - b||^2 + delta*||x||^2

    Closed-form solution in k-space:
        x = F^-1 * (1 / (mask + delta)) * b

    Parameters:
        kspace_undersampled : 2D array, undersampled k-space
        mask_1d             : 1D boolean array of length ky
        delta               : Regularization strength

    Returns:
        Reconstructed image (2D complex array)
    """
    ky, kx = kspace_undersampled.shape
    mask_2d = np.tile(mask_1d[:, np.newaxis].astype(np.float32), (1, kx))

    weights = 1.0 / (mask_2d + delta)

    kspace_weighted = weights * kspace_undersampled
    return transform(kspace_weighted), kspace_weighted, delta

def simulation():
    """

    :return:
    """
    deltas = [0.001, 0.25, 1]
    colors = ['royalblue', 'hotpink', 'lime']
    simuleringer = 200
    center_fractions = np.linspace(0, 1, simuleringer)

    plt.figure(figsize=(10, 6))

    for delta,c in zip(deltas,colors):
        rmse = np.zeros(simuleringer)
        percentage = np.zeros(simuleringer)

        for i, cf in enumerate(center_fractions):
            mask, _ = samplingmask(ky_size, center_fraction=cf)
            kspace_undersampled = kspace.copy()
            kspace_undersampled[~mask, :] = 0

            image_L2, _, _ = Leastsquares(kspace_undersampled, mask, delta)
            rmse[i] = RelativeMeanSquareError(image, image_L2)
            percentage[i] = mask.sum() / ky_size * 100

        plt.plot(percentage, rmse, label=f'δ = {delta}', color=c)

    plt.ylabel('RMSE')
    plt.xlabel('Sampling Percentage')
    plt.xticks(np.arange(0, 101, 5))
    plt.legend(title = "Chosen deltas")
    plt.grid(True, which='both')
    plt.show()

simulation()
