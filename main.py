from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import math, cmath, os
from typing import Optional
import requests
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Transmission Line Calculator (Updated)")
# IMAGE DICTIONARY (mapping user selection → image file)
IMAGE_MAP = {
    1: "Three Phase Single Circuit unbundled Triangular geometry",
    2: "Three Phase Single Circuit unbundled Colinear geometry",
    3: "Three Phase Double Circuit unbundled Vertical geometry",
    4: "Three Phase Double Circuit unbundled Staggered geometry",
    5: "Three Phase Single Circuit Bundled Triangular geometry (triangular bundled)",
    6: "Three Phase Single Circuit Bundled Triangular geometry (square bundled)",
    7: "Three Phase Single Circuit Bundled Triangular geometry (horizontal bundled)",
    8: "Three Phase Single Circuit Bundled Colinear geometry (horizontal bundled)",
    9: "Three Phase Single Circuit Bundled Colinear geometry (square bundled)",
    10: "Three Phase Single Circuit Bundled Colinear geometry (triangular bundled)",
    11: "Three Phase Double Circuit Bundled Vertical geometry (horizontal bundled)",
    12: "Three Phase Double Circuit Bundled Vertical geometry (square bundled)",
    13: "Three Phase Double Circuit Bundled Vertical geometry (triangular bundled)",
    14: "Three Phase Double Circuit Bundled Staggered geometry (horizontal bundled)",
    15: "Three Phase Double Circuit Bundled Staggered geometry (triangular bundled)",
    16: "Three Phase Double Circuit Bundled Staggered geometry (square bundled)",
}
from fastapi.staticfiles import StaticFiles

app.mount("/images", StaticFiles(directory="images"), name="images")


# Directories
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


def get_image_number(geometry_type, circuit, bundling):
    """
    geometry_type: 'triangular', 'colinear', 'vertical', 'staggered'
    circuit: 'single' or 'double'
    bundling: 'unbundled', 'horizontal', 'square', 'triangular'
    """

    # --- UNBUNDLED CASES ---
    if bundling == "unbundled":
        if circuit == "single":
            return 1 if geometry_type == "triangular" else 2
        if circuit == "double":
            return 3 if geometry_type == "vertical" else 4

    # --- BUNDLED CASES FOR SINGLE CIRCUIT ---
    if circuit == "single":
        if geometry_type == "triangular":
            return { "triangular": 5, "square": 6, "horizontal": 7 }[bundling]
        if geometry_type == "colinear":
            return { "horizontal": 8, "square": 9, "triangular": 10 }[bundling]

    # --- BUNDLED CASES FOR DOUBLE CIRCUIT ---
    if circuit == "double":
        if geometry_type == "vertical":
            return { "horizontal": 11, "square": 12, "triangular": 13 }[bundling]
        if geometry_type == "staggered":
            return { "horizontal": 14, "triangular": 15, "square": 16 }[bundling]

    return None




class TransmissionLineInductance:
    def __init__(self):
        # Updated ACSR conductor data (Ω/1000ft values from instructor)
        # radius_ft stored in feet (we'll convert to meters where needed)
        self.conductor_data = {
            "Dove":      {"resistance_ohm_per_1000ft": 0.1826, "radius_ft": 0.0314},
            "Drake":     {"resistance_ohm_per_1000ft": 0.1284, "radius_ft": 0.0373},
            "Partridge": {"resistance_ohm_per_1000ft": 0.3792, "radius_ft": 0.0217},
            "Ostrich":   {"resistance_ohm_per_1000ft": 0.3372, "radius_ft": 0.0229},
            "Hawk":      {"resistance_ohm_per_1000ft": 0.2120, "radius_ft": 0.0289},
            "Rook":      {"resistance_ohm_per_1000ft": 0.1603, "radius_ft": 0.0327},
            "Finch":     {"resistance_ohm_per_1000ft": 0.0937, "radius_ft": 0.0436},
            "Pheasant":  {"resistance_ohm_per_1000ft": 0.0821, "radius_ft": 0.0466},
            "Falcon":    {"resistance_ohm_per_1000ft": 0.0667, "radius_ft": 0.0523},
            "Custom":    {}
        }

    # --- basic formulas (expect D_eq and r_eq in meters) ---
    def calculate_inductance(self, D_eq, r_eq):
        return 2e-7 * math.log(D_eq / r_eq)

    def calculate_capacitance(self, D_eq, r_eq):
        epsilon = 8.854e-12
        return (2 * math.pi * epsilon) / math.log(D_eq / r_eq)

    # ===========================
    # SINGLE CIRCUIT
    # ===========================
    def case_single_colinear(self, D_ab_m, D_bc_m, D_ca_m, r_m, bundled=False, r_bundled_m=None, r_bundled_c_m=None):
        D_eq = (D_ab_m * D_bc_m * D_ca_m) ** (1 / 3)
        # for inductance instructor used factor 0.77 when not bundled; r_eq in instructor code used r*0.77
        r_eq_L = r_bundled_m if bundled and r_bundled_m else (r_m * 0.77)
        # capacitance uses plain conductor radius (or r_bundled_c if bundled)
        r_eq_C = r_bundled_c_m if bundled and r_bundled_c_m else r_m
        L = self.calculate_inductance(D_eq, r_eq_L)
        C = self.calculate_capacitance(D_eq, r_eq_C)
        return L, C

    def case_single_triangular(self, D_ab_m, D_bc_m, D_ca_m, r_m, bundled=False, r_bundled_m=None, r_bundled_c_m=None):
        # same approach as colinear
        D_eq = (D_ab_m * D_bc_m * D_ca_m) ** (1 / 3)
        r_eq_L = r_bundled_m if bundled and r_bundled_m else (r_m * 0.77)
        r_eq_C = r_bundled_c_m if bundled and r_bundled_c_m else r_m
        L = self.calculate_inductance(D_eq, r_eq_L)
        C = self.calculate_capacitance(D_eq, r_eq_C)
        return L, C

    # ===========================
    # DOUBLE CIRCUIT
    # ===========================
    def case_double_aligned(self, D_ac_prime_m, D_bb_prime_m, D_ac_m, r_m, bundled=False, r_bundled_m=None, r_bundled_c_m=None):
        # Derived distances (from instructor code)
        D_aa_prime = math.sqrt(D_ac_prime_m**2 + D_ac_m**2)
        D_ab_prime = math.sqrt(D_ac_prime_m**2 + (D_ac_m / 2)**2)

        # Equivalent distances used by instructor (converted to meters already)
        DAB_m = DBC_m = (( (D_ac_m / 2)**2 * D_ab_prime**2 ))**0.25
        DCA_m = ((D_ac_m**2 * D_ac_prime_m**2))**0.25

        D_eq = (DAB_m * DBC_m * DCA_m) ** (1/3)

        r_eq_L = r_bundled_m if bundled and r_bundled_m else (r_m * 0.77)
        r_eq_C = r_bundled_c_m if bundled and r_bundled_c_m else r_m

        # intermediate Ds for inductance / capacitance (instructor formulas)
        D_aa_prime_L = D_cc_prime_L = ((D_aa_prime**2 * r_eq_L**2))**0.25
        D_bb_prime_L = ((D_ac_prime_m**2 * r_eq_L**2))**0.25

        D_aa_prime_C = D_cc_prime_C = ((D_aa_prime**2 * r_eq_C**2))**0.25
        D_bb_prime_C = ((D_ac_prime_m**2 * r_eq_C**2))**0.25

        Ds_L = (D_aa_prime_L * D_cc_prime_L * D_bb_prime_L)**(1/3)
        Ds_C = (D_aa_prime_C * D_cc_prime_C * D_bb_prime_C)**(1/3)

        L = self.calculate_inductance(D_eq, Ds_L)
        C = self.calculate_capacitance(D_eq, Ds_C)
        return L, C

    def case_double_staggered(self, D_ac_prime_m, D_bb_prime_m, D_ac_m, r_m, bundled=False, r_bundled_m=None, r_bundled_c_m=None):
        # offset and derived distances
        offset_m = (D_bb_prime_m - D_ac_prime_m) / 2.0
        D_aa_prime = math.sqrt(D_ac_prime_m**2 + D_ac_m**2)
        D_ab = math.sqrt((D_ac_m / 2.0)**2 + offset_m**2)
        D_ab_prime = math.sqrt((D_ac_m / 2.0)**2 + (D_bb_prime_m - offset_m)**2)

        DAB_m = DBC_m = ((D_ab * D_ab_prime * D_ab_prime * D_ab))**0.25
        DCA_m = ((D_ac_m**2 * D_ac_prime_m**2))**0.25

        D_eq = (DAB_m * DBC_m * DCA_m) ** (1/3)

        r_eq_L = r_bundled_m if bundled and r_bundled_m else (r_m * 0.77)
        r_eq_C = r_bundled_c_m if bundled and r_bundled_c_m else r_m

        D_aa_prime_L = D_cc_prime_L = ((D_aa_prime**2 * r_eq_L**2))**0.25
        D_bb_prime_L = ((D_bb_prime_m**2 * r_eq_L**2))**0.25

        D_aa_prime_C = D_cc_prime_C = ((D_aa_prime**2 * r_eq_C**2))**0.25
        D_bb_prime_C = ((D_bb_prime_m**2 * r_eq_C**2))**0.25

        Ds_L = (D_aa_prime_L * D_cc_prime_L * D_bb_prime_L)**(1/3)
        Ds_C = (D_aa_prime_C * D_cc_prime_C * D_bb_prime_C)**(1/3)

        L = self.calculate_inductance(D_eq, Ds_L)
        C = self.calculate_capacitance(D_eq, Ds_C)
        return L, C

    # ===========================
    # PERFORMANCE (returns JSON-safe dict)
    # ===========================
    def performance_calculations(self, L_per_m, C_per_m, R_per_mile, length_km, freq_hz, V_L_kV, P_MW, pf, lagging=True):
        # Convert and compute
        length_miles = length_km * 0.621371
        length_m = length_km * 1000.0
        omega = 2.0 * math.pi * freq_hz

        R_total = R_per_mile * length_miles
        X_L = omega * L_per_m * length_m
        X_C = 1.0 / (omega * C_per_m * length_m) if C_per_m > 0 else float('inf')

        # Receiving end
        V_R_phase = (V_L_kV * 1000.0) / math.sqrt(3)
        S_R_phase = (P_MW * 1e6) / 3.0
        I_R_mag = S_R_phase / (V_R_phase * pf)
        phi = math.acos(min(max(pf, -1.0), 1.0))
        I_R = cmath.rect(I_R_mag, -phi if lagging else phi)

        Z = complex(R_total, X_L)
        Y = complex(0.0, 1.0 / X_C) if X_C != float('inf') else complex(0.0, 0.0)

        A = 1.0 + (Y * Z / 2.0)
        D = A
        B = Z
        C_mat = Y * (1.0 + (Y * Z / 4.0))

        V_S = A * V_R_phase + B * I_R
        I_S = C_mat * V_R_phase + D * I_R

        V_S_mag = abs(V_S)
        V_S_line_kV = (V_S_mag * math.sqrt(3.0)) / 1000.0
        I_S_mag = abs(I_S)

        V_R_no_load = abs(V_S / A) if A != 0 else 0.0
        voltage_reg = ((V_R_no_load - V_R_phase) / V_R_phase) * 100.0 if V_R_phase != 0 else 0.0

        S_S = 3.0 * V_S * I_S.conjugate()
        P_S_MW = S_S.real / 1e6
        Q_S_MVAR = S_S.imag / 1e6
        Apparent_MVA = abs(S_S) / 1e6
        pf_sending = P_S_MW / Apparent_MVA if Apparent_MVA != 0 else 0.0
        efficiency_percent = (P_MW / P_S_MW) * 100.0 if P_S_MW != 0 else 0.0

        return {
            "R_total_ohm": R_total,
            "X_L_ohm": X_L,
            "X_C_ohm": X_C,
            "V_S_phase_V": {"real": V_S.real, "imag": V_S.imag},
            "V_S_line_kV": V_S_line_kV,
            "V_S_LL_no_load_kV": (math.sqrt(3.0) * V_R_no_load) / 1000.0 if V_R_no_load else 0.0,
            "I_S_A": {"real": I_S.real, "imag": I_S.imag},
            "I_S_mag_A": I_S_mag,
            "voltage_reg_percent": voltage_reg,
            "Ps_MW": P_S_MW,
            "Qs_MVAR": Q_S_MVAR,
            "Apparent_MVA": Apparent_MVA,
            "pf_sending": pf_sending,
            "efficiency_percent": efficiency_percent
        }


calc = TransmissionLineInductance()


# -----------------------------------------------------------------------------------------
# Request Model
# -----------------------------------------------------------------------------------------
class CalculateRequest(BaseModel):
    circuit_type: int = Field(..., ge=1, le=2)
    geometry_subtype: int = Field(..., ge=1, le=2)
    d1: float = Field(..., gt=0)   # interpret per mapping confirmed
    d2: float = Field(..., gt=0)
    d3: float = Field(..., gt=0)
    bundled: bool = False
    bundle_shape: Optional[int] = Field(default=None, ge=1, le=3)   # 1=horizontal,2=triangular,3=square
    bundle_spacing_ft: Optional[float] = Field(default=None)

    conductor_type: str = Field(..., pattern="^(Dove|Drake|Partridge|Ostrich|Hawk|Rook|Finch|Pheasant|Falcon|Custom)$")
    custom_resistance: float = Field(..., gte=0)
    custom_radius: float = Field(..., gte=0)
    length_km: float = Field(..., gt=0)
    freq_hz: float = Field(..., gt=0)
    V_L_kV: float = Field(..., gt=0)
    P_MW: float = Field(..., gt=0)
    pf: float = Field(..., gt=0, le=1)
    pf_lagging: bool = True


# -----------------------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/calculate")
async def calculate(payload: CalculateRequest):
    # conductor selection
    if payload.conductor_type not in calc.conductor_data:
        return JSONResponse(status_code=400, content={"error": "Unknown conductor type"})

    if payload.conductor_type == "Custom":
        cond = {
            "resistance_ohm_per_1000ft": payload.custom_resistance,
            "radius_ft": payload.custom_radius
        }

    else:
        cond = calc.conductor_data[payload.conductor_type]
    print(cond["radius_ft"])
    # convert resistance from ohm per 1000 ft -> ohm per mile
    R_per_mile = cond["resistance_ohm_per_1000ft"]
    radius_ft = cond["radius_ft"]
    radius_m = radius_ft * 0.3048

    # bundled conductor equivalent radii (in meters)
    r_bundled_m = None
    r_bundled_c_m = None
    if payload.bundled and payload.bundle_spacing_ft and payload.bundle_shape:
        spacing = payload.bundle_spacing_ft
        # follow instructor's run_calculator formulas (spacing in ft, r in ft)
        r_ft = radius_ft
        if payload.bundle_shape == 1:
            # Horizontal (instructor formula: sqrt(r*0.77 * spacing) )
            r_bundled_ft = (r_ft * 0.77 * (spacing)) ** 0.5
            r_bundled_c_ft = (r_ft * (spacing)) ** 0.5
        elif payload.bundle_shape == 2:
            # Triangular
            r_bundled_ft = (r_ft * 0.77 * (spacing ** 2)) ** (1/3)
            r_bundled_c_ft = (r_ft * (spacing ** 2)) ** (1/3)
        else:
            # Square (instructor used a 1.0905 coefficient)
            r_bundled_ft = (1.0905 * (spacing**3 * r_ft * 0.77)) ** (1/4)
            r_bundled_c_ft = (1.0905 * (spacing**3 * r_ft)) ** (1/4)

        r_bundled_m = r_bundled_ft * 0.3048
        r_bundled_c_m = r_bundled_c_ft * 0.3048

    # Convert distances from ft -> m (frontend supplies ft)
    D1_m = payload.d1 * 0.3048
    D2_m = payload.d2 * 0.3048
    D3_m = payload.d3 * 0.3048

    # choose appropriate geometry function (mapping you confirmed)
    if payload.circuit_type == 1:
        if payload.geometry_subtype == 1:
            L_per_m, C_per_m = calc.case_single_colinear(D1_m, D2_m, D3_m, radius_m, payload.bundled, r_bundled_m, r_bundled_c_m)
        else:
            L_per_m, C_per_m = calc.case_single_triangular(D1_m, D2_m, D3_m, radius_m, payload.bundled, r_bundled_m, r_bundled_c_m)
    else:
        if payload.geometry_subtype == 1:
            L_per_m, C_per_m = calc.case_double_aligned(D1_m, D2_m, D3_m, radius_m, payload.bundled, r_bundled_m, r_bundled_c_m)
        else:
            L_per_m, C_per_m = calc.case_double_staggered(D1_m, D2_m, D3_m, radius_m, payload.bundled, r_bundled_m, r_bundled_c_m)

    perf = calc.performance_calculations(
        L_per_m, C_per_m, R_per_mile,
        payload.length_km, payload.freq_hz, payload.V_L_kV,
        payload.P_MW, payload.pf, payload.pf_lagging
    )
    # Determine circuit string
    circuit_str = "single" if payload.circuit_type == 1 else "double"

    # Determine geometry_type string
    if payload.circuit_type == 1:
        geometry_str = "triangular" if payload.geometry_subtype == 2 else "colinear"
    else:
        geometry_str = "vertical" if payload.geometry_subtype == 1 else "staggered"

    # Determine bundling string
    if not payload.bundled:
        bundling_str = "unbundled"
    else:
        bundling_map = {1: "horizontal", 2: "triangular", 3: "square"}
        bundling_str = bundling_map.get(payload.bundle_shape, "horizontal")  # default

    # Now get image number
    image_num = get_image_number(geometry_str, circuit_str, bundling_str)
    image_url = f"/images/{image_num}.png"


    response = {
        "L_per_m_H": L_per_m,
        "C_per_m_F": C_per_m,
        "R_per_mile_ohm": R_per_mile,
        "conductor_radius_ft": radius_ft,
        "performance": perf,
        "image_url": image_url
    }

    return JSONResponse(content=response)


