"""Small installed-package smoke test for the conda recipe."""

import importlib
from importlib.metadata import version

import numpy as np
import pandas as pd

import molecular_dynamics_tools as mdt

for dependency in ("MDAnalysis", "freud"):
    importlib.import_module(dependency)

assert mdt.__version__ == version("molecular-dynamics-tools")

r = np.linspace(0.0, 8.0, 161)
peak = np.exp(-((r - 2.5) / 0.4) ** 2)
rdfs = pd.DataFrame(
    {
        "r": r,
        "Cl-Cl": np.ones_like(r),
        "Cl-Na": 1.0 + 0.4 * peak,
        "Na-Na": np.ones_like(r),
    }
)
result = mdt.compute_scattering(
    rdfs,
    composition={"Cl": 2, "Na": 2},
    number_density=0.05,
    q_range=(0.0, 10.0),
    q_step=0.1,
)
for probe in (result.neutron, result.xray):
    assert probe is not None
    sf = probe.structure_factor
    rdf = probe.weighted_rdf
    assert np.isfinite(sf.to_numpy()).all()
    assert np.isfinite(rdf.to_numpy()).all()
    assert not np.allclose(sf["Total"], 1.0)
    np.testing.assert_allclose(
        sf["Total"], 1.0 + sf.drop(columns=["Q", "Total"]).sum(axis=1)
    )
