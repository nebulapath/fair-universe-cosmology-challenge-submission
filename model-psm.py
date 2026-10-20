from __future__ import annotations
import numpy as np
from pathlib import Path

import torch
import pickle
from torch import nn
from torch import optim
from scipy.ndimage import maximum_filter
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, InverseTransform, RandomPermutation
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import maximum_filter

HERE = Path(__file__).resolve().parent

class Model:
    def __init__(self) -> None:
        """
        Model class constructor
        """
        
        self.mask = np.load(HERE / "WIDE12H_bin2_2arcmin_mask.npy", allow_pickle=False)
        self.labels = np.load(HERE / "label_newrealization.npy", mmap_mode="r", allow_pickle=False)
        self.Ncosmo = 101
        self.Nsys = 256
        self.Ntest = 10000  # Number of instances in the test data
        self.pixelsize_arcmin = 2 # pixel size in arcmin
        self.pixelsize_radian = self.pixelsize_arcmin / 60 / 180 * np.pi # pixel size in radian

    def power_spectrum(x, pixsize, kedge):
        """
        Compute the azimuthally averaged 2D power spectrum of a real-valued 2D field.

        Parameters:
        -----------
        x : 2D numpy array
            Input real-space map (e.g., an image or simulated field).
            Must be a 2D array with shape (N_y, N_x).

        pixsize : float
            Physical size of each pixel in the map (e.g., arcmin, Mpc, etc.).
            Units should be consistent with the units used for `kedge`.

        kedge : 1D array-like
            Bin edges in wavenumber space (k), used to bin the power spectrum.
            Should be monotonically increasing and cover the k-range of interest.

        Returns:
        --------
        power_k : 1D numpy array
            The average wavenumber in each k bin (excluding the DC bin).

        power : 1D numpy array
            The binned, azimuthally averaged power spectrum corresponding to `power_k`.
            Normalized per unit area.
        """

        # Ensure the input array is 2D
        assert x.ndim == 2

        # Compute the 2D FFT of the input map and take its squared magnitude (power spectrum)
        xk = np.fft.rfft2(x)  # Real-to-complex FFT (along last axis)
        xk2 = (xk * xk.conj()).real  # Power spectrum: |FFT|^2

        # Get the shape of the input map
        Nmesh = x.shape

        # Compute the wavenumber grid (k-space)
        k = np.zeros((Nmesh[0], Nmesh[1]//2+1))
        # Square of the frequency in the first axis
        k += np.fft.fftfreq(Nmesh[0], d=pixsize).reshape(-1, 1) ** 2
        # Square of the frequency in the second axis (real FFT)
        k += np.fft.rfftfreq(Nmesh[1], d=pixsize).reshape(1, -1) ** 2
        # Convert from (1/length)^2 to angular frequency in radian units
        k = k ** 0.5 * 2 * np.pi

        # Bin each k value according to the bin edges provided in kedge
        index = np.searchsorted(kedge, k)

        # Bin the power values, number of modes, and wavenumbers
        power = np.bincount(index.flatten(), weights=xk2.flatten())
        Nmode = np.bincount(index.flatten())
        power_k = np.bincount(index.flatten(), weights=k.flatten())

        # Adjust for symmetry in the real FFT: include the mirrored part (excluding Nyquist frequency)
        if Nmesh[1] % 2 == 0:  # Even number of columns
            power += np.bincount(index[...,1:-1].flatten(), weights=xk2[...,1:-1].flatten())
            Nmode += np.bincount(index[...,1:-1].flatten())
            power_k += np.bincount(index[...,1:-1].flatten(), weights=k[...,1:-1].flatten())
        else:  # Odd number of columns
            power += np.bincount(index[...,1:].flatten(), weights=xk2[...,1:].flatten())
            Nmode += np.bincount(index[...,1:].flatten())
            power_k += np.bincount(index[...,1:].flatten(), weights=k[...,1:].flatten())

        # Exclude the first bin (typically corresponds to DC mode)
        power = power[1:len(kedge)]
        Nmode = Nmode[1:len(kedge)]
        power_k = power_k[1:len(kedge)]

        # Average the power and wavenumber in each bin, only where Nmode > 0
        select = Nmode > 0
        power[select] = power[select] / Nmode[select]
        power_k[select] = power_k[select] / Nmode[select]

        # Normalize the power spectrum by the map area
        power *= pixsize ** 2 / Nmesh[0] / Nmesh[1]

        # Return the binned k values and corresponding power spectrum
        return power_k, power
    
    def extract_peak_counts(map_2d, thresholds=np.linspace(-0.02, 0.06, 8)):
        """Extracts the local peak counts above different threshold levels."""
        local_max = (maximum_filter(map_2d, size=3) == map_2d)
        peaks = map_2d[local_max]
        counts, _ = np.histogram(peaks, bins=thresholds)
        return counts

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """
        Params:
            test_data (np.ndarray): An array containing the test data
                                    shape = (N, 132019), where N is the number of samples in the test dataset

        Functionality:
            This function can be used for predictions using the test sets

            To apply the mask and produce 2D maps, you can include the mask .npy file in model_dir (the submission folder) and load it here.
            # Example:
                mask = np.load(os.path.join(model_dir, "WIDE12H_bin2_2arcmin_mask.npy"))
                N = len(test_data)
                test_maps = np.zeros((N, 1424, 176), dtype=np.float16)
                test_maps[:, mask] = test_data

        Returns:
            A numpy array of the predicted OoD scores
            shape = (N,)
        """

        Nbins = 10
        l_edge = np.logspace(2, 4, Nbins+1)
        
        with open("scaler.pkl", "rb") as f:
            scaler_joint = pickle.load(f)
    
        scaled_features = np.load(HERE / "joint_features_scaled.npy", allow_pickle=False)
        
        num_peaks_features = 7
        num_features = Nbins + num_peaks_features
        num_layers = 5

        base_dist = StandardNormal(shape=[num_features])
        transforms = []
        for _ in range(num_layers):
            transforms.append(MaskedAffineAutoregressiveTransform(features=num_features,
                                                                hidden_features=64,
                                                                context_features=None))
            transforms.append(RandomPermutation(features=num_features))

        transform = CompositeTransform(transforms)

        flow_er = Flow(transform, base_dist)
        flow_er.load_state_dict(torch.load('flow_model.pt'))

        flow_er.eval()
        
        # 3. Extract same joint features on Test Data
        print("Extracting joint summary statistics for test data...")
        test_joint_features = np.zeros((self.Ntest, Nbins + num_peaks_features))

        for i in range(self.Ntest):
            # Note: Test data already contains noise
            test_map = test_data[i].astype(np.float64)
            _, ps = self.power_spectrum(test_map, self.pixelsize_radian, l_edge)
            pc = self.extract_peak_counts(test_map)
            test_joint_features[i] = np.concatenate([np.log10(ps), pc])

        test_joint_scaled = scaler_joint.transform(test_joint_features)
        test_tensor = torch.tensor(test_joint_scaled, dtype=torch.float32)
        
        # 4. Evaluate OoD Scores via Negative Log-Likelihood from the flow
        print("Evaluating OoD scores on test set...")
        with torch.no_grad():
            # Evaluate log probabilities in batches to avoid GPU OOM
            log_probs = []
            test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_tensor), batch_size=1000, shuffle=False)
            for batch in test_loader:
                x_batch = batch[0]
                log_probs.append(flow_er.log_prob(x_batch).cpu().numpy())
            log_probs = np.concatenate(log_probs)

        # Higher negative log-probability = more likely OoD
        ood_scores = -log_probs
        print("OoD scores calculated! Ready to package and submit.")
  
        return ood_scores