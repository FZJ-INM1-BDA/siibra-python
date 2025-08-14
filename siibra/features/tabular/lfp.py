import numpy as np

from . import tabular

class LFP(tabular.Tabular, configuration_folder="features/tabular/lfp", category="cellular"):
    def __init__(self, zarr_group_url, description, modality, anchor, data, datasets = None, prerelease = False, id = None):
        super().__init__(description, modality, anchor, data, datasets, prerelease, id)
        self.zarr_group_url = zarr_group_url
    
    def plot(self, *args, backend="matplotlib", **kwargs):
        try:
            import plotly as py
            import matplotlib.pyplot as plt
            import zarr
            from scipy.ndimage import uniform_filter1d
        except ImportError as e:
            raise ImportError("Need python3.11 for LFP plot function") from e
        
        f = zarr.open_group(self.zarr_group_url, mode="r")

        spectrum_type = 'spectrogram_rhythmic'  # Options: 'spectrogram', 'spectrogram_rhythmic', 'spectrogram_arrhythmic'
        times = f['/times'][:]
        freqs = f['/frequencies'][:]

        if spectrum_type =='spectrogram':
            # Regular PSD spectrogram (Hann windowed, 50% overlap)
            spectrogram = f['/spectrogram'][:]
            spectrogram = 10 * np.log10(spectrogram)  # Convert to dB scale
        if spectrum_type =='spectrogram_rhythmic':
            # Use the rhythmic part of the spectrogram (IRASA; Wen H. and Liu Z., 2015)
            spectrogram = f['/spectrogram_rhythmic'][:] # Already in dB scale: 10*log10(S/S_arrhythmic)
        if spectrum_type =='spectrogram_arrhythmic':
            # Use the arrhythmic part of the spectrum (IRASA)
            spectrogram = f['/spectrogram_arrhythmic'][:]
            spectrogram = 10 * np.log10(spectrogram)  # Convert to dB scale

        # Flatten freqs and times to 1D
        freqs = freqs.flatten()
        times = times.flatten()

        
        # Median and percentiles (along the time axis)
        P_m = np.nanmedian(spectrogram, axis=0)
        P_25 = np.nanpercentile(spectrogram, 25, axis=0)
        P_75 = np.nanpercentile(spectrogram, 75, axis=0)

        # Replace line noise with average of the edges
        Freqs = [50, 100, 150, 200, 250, 300] # Line noise frequency and harmonics (in Europe)
        S_m = P_m.copy()  # Copy median spectrum
        for freq in Freqs:
            mask = (freqs > freq - 1) & (freqs < freq + 1) # Create mask +-1 Hz around the line noise frequency
            indices = np.where(mask)[0]
            average_value = np.nanmean([S_m[indices[0]], S_m[indices[-1]]]) # Average the values at the edges
            S_m[mask] = average_value
            
        # Smooth along the frequency axis
        S_m = uniform_filter1d(S_m, size=21, mode='nearest', origin=0)

        # Plot
        plt.style.use('dark_background')  # Set dark background
        plt.figure(1)
        plt.clf()
        plt.plot(freqs, P_m, color=[0.5, 0.5, 0.5], label='Median', alpha=0.4)
        plt.fill_between(freqs, P_25, P_75, color=[0.5, 0.5, 0.5], edgecolor='none', alpha=0.25, label='25-75th percentile')
        plt.plot(freqs, S_m, '#5E6ABA', linewidth=2, label='Smoothed median')
        plt.xlabel('Frequency (Hz)')
        plt.grid(axis='x', color='gray', linestyle='--', alpha=0.5)  # Add vertical grid lines

        if spectrum_type == "spectrogram":
            plt.ylabel('dB')
            plt.title('PSD')
        if spectrum_type == "spectrogram_rhythmic":
            plt.ylabel('dB(fractal)')
            plt.title(f"PSD (rhythmic part)")
        if spectrum_type == "spectrogram_arrhythmic":
            plt.ylabel('dB')
            plt.title('PSD (arrhythmic part)')
        
        plt.legend()
        plt.tight_layout()
        fig = plt.gcf()

        if backend == "matplotlib":
            return fig

        if backend == "plotly":
            pyfig = py.tools.mpl_to_plotly(fig)
            return py.offline.plot(pyfig)
