import numpy as np
import pandas as pd
from scipy.constants import h, c, e

# ----------------------------
# Input Parameters
# ----------------------------
Eg = 2.0951 # Heterostructure bandgap (eV) TYPE-II CdCl2/Zn2S2
EgA = 3.88  # Bandgap of material A (CdCl2) in eV
EgB = 2.53  # Bandgap of material B (Z2S2) in eV
CBM_A = -4.177  # Conduction band minimum of material A (eV vs vacuum) - CdCl2
VBM_B = -6.2723  # Valence band maximum of material B (eV vs vacuum) - Zn2S2
del_phi = 0.00015  # Electrostatic potential difference (eV)

# Eg = 2.603  # Heterostructure bandgap (eV) MoSi2N4
# EgA = Eg  # Bandgap of material A (GaSe) in eV
# EgB = Eg  # Bandgap of material B (ZrS2) in eV
# CBM_A = -3.5835  # Conduction band minimum of material A (eV vs vacuum)
# VBM_B = -6.1868  # Valence band maximum of material B (eV vs vacuum)
# del_phi = 0.00003  # Electrostatic potential difference (eV)

# Eg = 2.9408 # Heterostructure bandgap (eV) Ni3TeO6
# EgA = Eg  # Bandgap of material A (GaSe) in eV
# EgB = Eg  # Bandgap of material B (ZrS2) in eV
# CBM_A = -3.466  # Conduction band minimum of material A (eV vs vacuum)
# VBM_B = -6.4069  # Valence band maximum of material B (eV vs vacuum)
# del_phi = 0  # Electrostatic potential difference (eV)

# Eg = 2.22 # Heterostructure bandgap (eV) TYPE-II Ni3TeO6/MoSi2N4
# EgA = 3.10  # Bandgap of material A (GaSe) in eV
# EgB = 4.92  # Bandgap of material B (ZrS2) in eV
# CBM_A = -4.1844  # Conduction band minimum of material A (eV vs vacuum) - MSN
# VBM_B = -6.4038  # Valence band maximum of material B (eV vs vacuum) - NTO
# del_phi = 0.00085  # Electrostatic potential difference (eV)

# Eg = 1.9 # Heterostructure bandgap (eV) TYPE-II WSi2N4/MoSi2N4
# EgA = Eg  # Bandgap of material A (GaSe) in eV
# EgB = Eg  # Bandgap of material B (ZrS2) in eV
# CBM_A = -3.61  # Conduction band minimum of material A (eV vs vacuum) - MSN
# VBM_B = -5.74  # Valence band maximum of material B (eV vs vacuum) - WSN
# del_phi = 0.14  # Electrostatic potential difference (eV)

# Eg = 1.07 # Heterostructure bandgap (eV) Z-SCHEME InP/ZrS2
# EgA = 1.94  # Bandgap of material A (InP) in eV
# EgB = 1.82  # Bandgap of material B (ZrS2) in eV
# CBM_A = -3.96  # Conduction band minimum of material A (eV vs vacuum)
# VBM_B = -6.65  # Valence band maximum of material B (eV vs vacuum)
# del_phi = 0.58  # Electrostatic potential difference (eV)

# ----------------------------
# Constants
# ----------------------------
del_G_H = 1.23  # Gibbs free energy for water-splitting (eV)
E_H2 = -4.44  # H+/H2 potential (eV)
E_O2 = -5.67  # H2O/O2 potential (eV)

# ----------------------------
# Overpotential Calculations
# ----------------------------
χA = CBM_A - (E_H2 - del_phi)
χB = E_O2 - VBM_B

# ----------------------------
# Lowest photon energy Calculations
# ----------------------------
#-------------- TYPE-I or II --------------
if χA >= 0.2 and χB >= 0.6:
    E = Eg
elif χA < 0.2 and χB >= 0.6:
    E = Eg + 0.2 - χA
elif χA >= 0.2 and χB < 0.6:
    E = Eg + 0.6 - χB
else:
    E = Eg + 0.2 - χA + 0.6 - χB

#-------------- Z-SCHEME --------------
# if χA >= 0.2 and χB >= 0.6:
#     E = max(EgA, EgB)
# elif χA < 0.2 and χB >= 0.6:
#     E = max(EgA + (0.2 - χA), EgB)
# elif χA >= 0.2 and χB < 0.6:
#     E = max(EgA, EgB + (0.6 - χB))
# else:
#     E = max(EgA + (0.2 - χA), EgB + (0.6 - χB))

# ----------------------------
# Load AM1.5G data
# ----------------------------
data = pd.read_csv("AM1.5.csv", skiprows=2, names=["Wavelength_nm", "Irradiance_W_m2_nm"])

# ----------------------------
# Convert wavelength to photon_energy and solar_irradiance to flux
# ----------------------------
data["Photon_Energy_eV"] = 1240 / data["Wavelength_nm"] # eV·nm/nm
hc_eVnm = 1240 # eV·nm
# Use the squared wavelength to account for the energy interval conversion (Jacobian term)
data["Photon_Flux"] = (data["Irradiance_W_m2_nm"] * data["Wavelength_nm"]**2) / (hc_eVnm * e) # photons/m²/s/eV
# data["Photon_Flux"] = (data["Irradiance_W_m2_nm"] * data["Wavelength_nm"]) / (hc_eVnm * e)
data = data.sort_values("Photon_Energy_eV", ascending=True)

# ----------------------------
# Masking data for integtals
# ----------------------------
data["Absorbed_Photon"] = data["Photon_Energy_eV"] >= Eg
data["Usable_Photon"] = data["Photon_Energy_eV"] >= E

# ----------------------------
# Calculate integrands for efficiency calculations
# ----------------------------
data["n_abs_Integrand"] = data["Photon_Flux"] * data["Absorbed_Photon"]
data["n_cu_Integrand"] = (data["Photon_Flux"] / data["Photon_Energy_eV"]) * data["Usable_Photon"]
data["phi_Integrand"] = (data["Photon_Flux"] / data["Photon_Energy_eV"]) * data["Absorbed_Photon"]

"""
# ----------------------------
# Efficiency Formulae
# ----------------------------
n_abs_num = ∫(Eg to ∞) (Photon_Flux ≥ Eg) d(Photon_Energy_eV)
n_abs_den = ∫(0 to ∞) (All Photon_Flux) d(Photon_Energy_eV)
----- FOR Z-SCHEME -----
n_cu_num = 0.5 * ΔG * ∫(E to ∞) (Photon_Flux / Photon_Energy_eV) d(Photon_Energy_eV)
----- FOR TYPE-I or II -----
n_cu_num = ΔG * ∫(E to ∞) (Photon_Flux / Photon_Energy_eV) d(Photon_Energy_eV)
n_cu_den = ∫(Eg to ∞) (Photon_Flux / Photon_Energy_eV) d(Photon_Energy_eV)

n_STH = n_abs * n_cu

phi_term = del_phi * ∫(Eg to ∞) (Photon_Flux / Photon_Energy_eV) d(Photon_Energy_eV)
n'_STH = n_STH * n_abs_den / (n_abs_den + phi_term)

# ----------------------------
# Absorption efficiency (n_abs)
# ----------------------------
Numerator: n_abs_num = ∫ (Photon_Flux ≥ Eg) → Integrates from Eg to 4000 nm (effectively ∞)
Denominator: n_abs_den = ∫ (All Photon_Flux) → Integrates from 280 nm to 4000 nm (≈ 0 to ∞)

# ----------------------------
# Carrier utilization efficiency (n_cu)
# ----------------------------
Numerator: n_cu_num = ∫ (Photon_Flux / Energy ≥ E) → Integrates from E to 4000 nm.
Denominator: n_cu_den → Same as η_abs numerator (from Eg to ∞)

# ----------------------------
# Corrected STH efficiency (n'_STH)
# ----------------------------
Denominator: n_abs_den + phi_term → Uses total flux (0 to ∞) and correction term (Eg to ∞)

"""

# ----------------------------
# Integrals using trapezoidal rule
# ----------------------------
n_abs_num = np.trapz(data[data["Absorbed_Photon"]]["Photon_Flux"], 
                    data[data["Absorbed_Photon"]]["Photon_Energy_eV"])
n_abs_den = np.trapz(data["Photon_Flux"], data["Photon_Energy_eV"])

# n_cu_num = 0.5 * del_G_H * np.trapz(data[data["Usable_Photon"]]["Photon_Flux"] / 
n_cu_num = del_G_H * np.trapz(data[data["Usable_Photon"]]["Photon_Flux"] / 
          data[data["Usable_Photon"]]["Photon_Energy_eV"], 
          data[data["Usable_Photon"]]["Photon_Energy_eV"])
phi_term = del_phi * np.trapz(data[data["Absorbed_Photon"]]["Photon_Flux"] / 
                           data[data["Absorbed_Photon"]]["Photon_Energy_eV"], 
                           data[data["Absorbed_Photon"]]["Photon_Energy_eV"])

# ----------------------------
# Efficiency Calculations
# ----------------------------
n_abs = n_abs_num / n_abs_den
n_cu = n_cu_num / n_abs_num
n_STH = n_abs * n_cu
n_STH_corr = n_STH * n_abs_den / (n_abs_den + phi_term)

# Sorting data back by wavelength
data_sorted_by_wavelength = data.sort_values("Wavelength_nm", ascending=True)

results = pd.DataFrame({
    "Parameter": [
        "χA(H2)", "χB(O2)", "Critical Energy (E)",
        "n_abs Numerator", "n_abs Denominator", "n_abs",
        "n_cu Numerator", "n_cu Denominator", "n_cu",
        "phi_term", "n_STH", "n'_STH"
    ],
    "Value": [
        f"{χA:.2f} V", f"{χB:.2f} V", f"{E:.2f} eV",
        f"{n_abs_num:.4e}", f"{n_abs_den:.4e}", f"{n_abs*100:.2f}%",
        f"{n_cu_num:.4e}", f"{n_abs_num:.4e}", f"{n_cu*100:.2f}%",
        f"{phi_term:.4e}", f"{n_STH*100:.2f}%", f"{n_STH_corr*100:.2f}%"
    ],
    "Description": [
        "HER overpotential", "OER overpotential", 
        "Minimum photon energy for water splitting",
        "Absorbed photon flux integral", "Total photon flux integral",
        "Light absorption efficiency",
        "Usable carrier flux integral", "Absorbed photon flux integral",
        "Carrier utilization efficiency",
        "Polarization correction term", 
        "Solar-to-Hydrogen efficiency",
        "Corrected STH efficiency"
    ]
})

data_sorted_by_wavelength.to_csv("detailed_calculations.csv", index=False)
results.to_csv("efficiency_results.csv", index=False)

print("\nFinal Results:")
print(results.to_string(index=False))
