"""Read config.yaml and resolve the paths and derived constants from it.

Every script starts with `cfg = config.load()`.  Anything a script would
otherwise have written at the top as a module constant lives in config.yaml
instead, so a threshold appears exactly once in the repository.

The few quantities that follow from those parameters -- the magnitude of the
flux ceiling, the enclosed-flux fraction of the photometric aperture -- are
computed here rather than typed in anywhere.
"""

import os

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Config(dict):
    def _dir(self, key):
        path = os.path.expanduser(self["paths"][key])
        if not os.path.isabs(path):
            path = os.path.normpath(os.path.join(ROOT, path))
        return path

    def data(self, key):
        """Absolute path to one of the input catalogues or images."""
        return os.path.join(self._dir("data"), self["inputs"][key])

    def out(self, name=""):
        return os.path.join(self._make("outputs"), name)

    def fig(self, name=""):
        return os.path.join(self._make("figures"), name)

    def table(self, name=""):
        return os.path.join(self._make("tables"), name)

    def result(self, name=""):
        return os.path.join(self._make("results"), name)

    def inspect(self, name=""):
        return os.path.join(self._make("inspect"), name)

    def _make(self, key):
        path = self._dir(key)
        os.makedirs(path, exist_ok=True)
        return path

    # --- quantities derived from the configured parameters ------------------

    def distance_cm(self):
        return self["distance"]["lmc_kpc"] * 1e3 * self["distance"]["parsec_cm"]

    def magnitude(self, flux_jy):
        """Radio absolute magnitude of a flux density at the LMC distance."""
        luminosity = 4 * np.pi * self.distance_cm() ** 2 * np.asarray(flux_jy) * 1e-23
        return -2.5 * np.log10(luminosity / self["pnlf"]["reference_luminosity_cgs"])

    def ceiling_magnitude(self):
        """M_radio of the flux ceiling.  The bright mask of the PNLF fit."""
        return float(self.magnitude(self["flux_ceiling"]["ceiling_mjy"] * 1e-3))

    def completeness_flux_ujy(self):
        """Flux density corresponding to the adopted completeness limit."""
        m_lim = self["pnlf"]["completeness_limit_mag"]
        luminosity = self["pnlf"]["reference_luminosity_cgs"] * 10 ** (-m_lim / 2.5)
        return luminosity / (4 * np.pi * self.distance_cm() ** 2) * 1e23 * 1e6

    def enclosed_fraction(self, radius_arcsec=None):
        """Fraction of an unresolved source inside a circular aperture.

        For a Gaussian beam, 1 - exp(-4 ln2 r^2 / FWHM^2).  At r = FWHM/2 this
        is exactly 0.5, which is the factor of two the sub-threshold aperture
        photometry has to be corrected by.
        """
        if radius_arcsec is None:
            radius_arcsec = self["visual"]["aperture_arcsec"]
        fwhm = self["meerkat"]["beam_fwhm_arcsec"]
        return float(1.0 - np.exp(-4.0 * np.log(2) * radius_arcsec ** 2 / fwhm ** 2))

    def visual_ids(self):
        """The inspected sub-threshold detections.

        Returns a list of (RP_ID, search_radius), the radius being None where
        the file gives no override and the default search radius applies.
        """
        path = os.path.join(ROOT, self["visual"]["detections_file"])
        entries = []
        with open(path) as fh:
            for line in fh:
                fields = line.split("#")[0].split()
                if fields:
                    radius = float(fields[1]) if len(fields) > 1 else None
                    entries.append((fields[0], radius))
        return entries


def load(path=None):
    """Read config.yaml, or the file named by $PNLF_CONFIG if it is set.

    The environment variable is how a run is pointed at a different data
    directory, or a scratch output directory, without editing the repository's
    own configuration.
    """
    path = path or os.environ.get("PNLF_CONFIG") or os.path.join(ROOT, "config.yaml")
    with open(path) as fh:
        return Config(yaml.safe_load(fh))
