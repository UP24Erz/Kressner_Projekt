# ============================================================
# Teilaufgabe 2 – Implementierung (Python, PuLP/CBC)
# OHNE Nachtregel
# OHNE Tie-Break
# MIT Diesel-Spritpreis fix: 1,60 €/Liter
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime
import pulp

try:
    from IPython.display import display
except Exception:
    display = print

# ----------------------------
# 1) Daten einlesen
# ----------------------------
PATH_ROUTES   = r"C:/Users/umut-/Desktop/Fallstudie Kressner/routes.csv"
PATH_CHARGERS = r"C:\Users\umut-\Desktop\Fallstudie Kressner\chargers.csv"
PATH_DIESEL   = r"C:\Users\umut-\Desktop\Fallstudie Kressner\diesel_trucks.csv"
PATH_ELECTRIC = r"C:\Users\umut-\Desktop\Fallstudie Kressner\electric_trucks.csv"

def read_semicolon_csv(path):
    for enc in ["utf-8-sig", "utf-8", "ISO-8859-1", "cp1252"]:
        try:
            df = pd.read_csv(path, sep=";", encoding=enc)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    return pd.read_csv(path, sep=";", engine="python")

routes   = read_semicolon_csv(PATH_ROUTES)
chargers = read_semicolon_csv(PATH_CHARGERS)
diesel   = read_semicolon_csv(PATH_DIESEL)
electric = read_semicolon_csv(PATH_ELECTRIC)

print("ROUTES");   print(routes.head(), "\n")
print("CHARGERS"); print(chargers.head(), "\n")
print("DIESEL");   print(diesel.head(), "\n")
print("ELECTRIC"); print(electric.head(), "\n")

# ----------------------------
# 2) Parameter
# ----------------------------
N_DAYS = 260                          # Betriebstage/Jahr
TOLL_EUR_PER_KM = 0.34                # Maut €/km (nur Diesel)

# Netz / Strom
GRID_LIMIT_KW = 500
TRAFO_EXTRA_KW = 500
TRAFO_COST_YEAR = 10000

ENERGY_PRICE_EUR_PER_KWH = 0.25
GRID_BASE_FEE_YEAR = 1000
DEMAND_CHARGE_EUR_PER_KW_YEAR = 150   # Beispielwert (€/kW*a)

# Zeitdiskretisierung
TIME_STEP_MIN = 15
DT_H = TIME_STEP_MIN / 60
N_T = int(24 * 60 / TIME_STEP_MIN)

# Speicher (optional)
BESS_CAPEX_PER_KW_YEAR  = 30
BESS_CAPEX_PER_KWH_YEAR = 350
BESS_OPEX_SHARE = 0.02
BESS_ROUNDTRIP_EFF = 0.98
BESS_MIN_SOC_SHARE = 0.025

# Diesel-Spritpreis FIX
DIESEL_PRICE_EUR_PER_LITER = 1.60

# ----------------------------
# 3) Vorverarbeitung: Tourmasken a[r,t], b[r,t]
# ----------------------------
def parse_hhmm(s):
    return datetime.strptime(str(s), "%H:%M")

def time_to_index(t):
    return int((t.hour * 60 + t.minute) / TIME_STEP_MIN)

R = list(routes["route_id"].astype(str))
route_name = dict(zip(routes["route_id"].astype(str), routes["route_name"]))
dist_total = dict(zip(routes["route_id"].astype(str), routes["distance_total"]))
dist_toll  = dict(zip(routes["route_id"].astype(str), routes["distance_toll"]))

start_idx, end_idx = {}, {}
for _, row in routes.iterrows():
    r = str(row["route_id"])
    si = time_to_index(parse_hhmm(row["starttime"]))
    ei = time_to_index(parse_hhmm(row["endtime"]))
    if ei < si:
        raise ValueError(f"Tour {r} läuft über Mitternacht – Modell nicht dafür ausgelegt.")
    start_idx[r] = si
    end_idx[r]   = ei

a = {(r,t): 0 for r in R for t in range(N_T)}
b = {(r,t): 0 for r in R for t in range(N_T)}  # Tourende-Indikator
for r in R:
    for t in range(start_idx[r], end_idx[r]):  # Ende exklusiv => hintereinander möglich
        a[(r,t)] = 1
    b[(r, end_idx[r]-1)] = 1

print("Anzahl Touren:", len(R))

# ----------------------------
# 4) Parameter aus CSVs
# ----------------------------
# E-Trucks
E = list(electric["truck_model"].astype(str))
capex_e = dict(zip(E, electric["capex_yearly"]))
opex_e  = dict(zip(E, electric["opex_yearly"]))
thg_e   = dict(zip(E, electric["thg_yearly"]))
pveh_e  = dict(zip(E, electric["max_power"]))      # kW
batt_e  = dict(zip(E, electric["soc_max_kWh"]))    # kWh
cons_e_per_km = dict(zip(E, electric["avg_energy_kWh_per_100km"] / 100.0))

# Diesel
capex_d = float(diesel.loc[0, "capex_yearly"])
opex_d  = float(diesel.loc[0, "opex_yearly"])
kfz_d   = float(diesel.loc[0, "kfz_yearly"])

# Diesel-Verbrauch (l/100km -> l/km)
if "avg_diesel_per_100km" not in diesel.columns:
    raise ValueError("In diesel_trucks.csv fehlt die Spalte 'avg_diesel_per_100km'.")

diesel_l_per_km = float(diesel.loc[0, "avg_diesel_per_100km"]) / 100.0
diesel_fuel_cost_per_km = diesel_l_per_km * DIESEL_PRICE_EUR_PER_LITER

print(f"Diesel Verbrauch l/km: {diesel_l_per_km:.4f}")
print(f"Diesel Spritkosten €/km (bei {DIESEL_PRICE_EUR_PER_LITER:.2f} €/l): {diesel_fuel_cost_per_km:.4f}")

# Chargers
S = list(chargers["charger_model"].astype(str))
charger_capex = dict(zip(S, chargers["capex_yearly"]))
charger_opex  = dict(zip(S, chargers["opex_yearly"]))
charger_pmax  = dict(zip(S, chargers["max_power"]))           # kW
charger_spots = dict(zip(S, chargers["charging_spots"]))      # Ladepunkte

# Energieverbrauch pro Route (kWh) je E-Typ
energy_route = {(r,e): dist_total[r] * cons_e_per_km[e] for r in R for e in E}

# ----------------------------
# 5) MILP Modell
# ----------------------------
K = list(range(len(R)))  # Upper bound

model = pulp.LpProblem("ETruck_Fleet_Charging_NO_NIGHT_NO_TIEBREAK", pulp.LpMinimize)

# Variablen
uE = pulp.LpVariable.dicts("uE", (K, E), 0, 1, cat="Binary")
uD = pulp.LpVariable.dicts("uD", (K,),    0, 1, cat="Binary")

xE = pulp.LpVariable.dicts("xE", (K, R, E), 0, 1, cat="Binary")
xD = pulp.LpVariable.dicts("xD", (K, R),    0, 1, cat="Binary")

nS = pulp.LpVariable.dicts("nCharger", (S,), lowBound=0, cat="Integer")
zTrafo = pulp.LpVariable("zTrafo", 0, 1, cat="Binary")

# Laden: c=lädt (hier nicht "steckt"), p=Leistung
c = pulp.LpVariable.dicts("cCharge", (K, range(N_T)), 0, 1, cat="Binary")
p = pulp.LpVariable.dicts("pChargeKW", (K, range(N_T)), lowBound=0, cat="Continuous")

SOC = pulp.LpVariable.dicts("SOCkWh", (K, range(N_T+1)), lowBound=0, cat="Continuous")

g = pulp.LpVariable.dicts("GridKW", (range(N_T),), lowBound=0, cat="Continuous")
Gmax = pulp.LpVariable("Gmax", lowBound=0, cat="Continuous")

# Speicher
Pbat = pulp.LpVariable("Pbat_kW", lowBound=0, cat="Continuous")
Ebat = pulp.LpVariable("Ebat_kWh", lowBound=0, cat="Continuous")
pch  = pulp.LpVariable.dicts("BESS_ch_kW",  (range(N_T),), lowBound=0, cat="Continuous")
pdis = pulp.LpVariable.dicts("BESS_dis_kW", (range(N_T),), lowBound=0, cat="Continuous")
Eb   = pulp.LpVariable.dicts("BESS_E_kWh",  (range(N_T+1),), lowBound=0, cat="Continuous")

# Helper: fährt k in t?
def driving_expr(k, t):
    return (
        pulp.lpSum(a[(r,t)] * xD[k][r] for r in R)
        + pulp.lpSum(a[(r,t)] * xE[k][r][e] for r in R for e in E)
    )

# ----------------------------
# Nebenbedingungen
# ----------------------------

# (1) Jede Route genau einmal
for r in R:
    model += (
        pulp.lpSum(xD[k][r] for k in K) + pulp.lpSum(xE[k][r][e] for k in K for e in E) == 1
    ), f"RouteOnce_{r}"

# (2) Pro Fahrzeug höchstens ein Typ + Nutzung nur wenn gekauft
for k in K:
    model += (uD[k] + pulp.lpSum(uE[k][e] for e in E) <= 1), f"OneType_{k}"
    for r in R:
        model += (xD[k][r] <= uD[k]), f"UseDieselOnlyIfBought_{k}_{r}"
        for e in E:
            model += (xE[k][r][e] <= uE[k][e]), f"UseEOnlyIfBought_{k}_{r}_{e}"

# (3) Keine parallelen Touren
for k in K:
    for t in range(N_T):
        model += (driving_expr(k,t) <= 1), f"NoOverlap_{k}_{t}"

# (4) Laden nur wenn E-Fahrzeug & nicht während Fahrt
Mbig = 10_000
for k in K:
    eBought = pulp.lpSum(uE[k][e] for e in E)
    for t in range(N_T):
        driving = driving_expr(k,t)
        model += (c[k][t] <= eBought), f"ChargeOnlyIfE_{k}_{t}"
        model += (c[k][t] + driving <= 1), f"NoDriveAndCharge_{k}_{t}"
        model += (p[k][t] <= Mbig * c[k][t]), f"p_le_c_{k}_{t}"
        model += (p[k][t] <= pulp.lpSum(pveh_e[e] * uE[k][e] for e in E)), f"p_le_Pveh_{k}_{t}"

# (5) Charger-Kapazitäten + max 3 Säulen
model += (pulp.lpSum(nS[s] for s in S) <= 3), "Max3Chargers"
for t in range(N_T):
    model += (pulp.lpSum(c[k][t] for k in K) <= pulp.lpSum(charger_spots[s] * nS[s] for s in S)), f"SpotsCap_{t}"
    model += (pulp.lpSum(p[k][t] for k in K) <= pulp.lpSum(charger_pmax[s] * nS[s] for s in S)), f"PowerCap_{t}"

# (6) Netzlimit + Trafo
for t in range(N_T):
    model += (g[t] <= GRID_LIMIT_KW + TRAFO_EXTRA_KW * zTrafo), f"GridLimit_{t}"

# (7) Leistungsbilanz
for t in range(N_T):
    model += (pulp.lpSum(p[k][t] for k in K) + pch[t] == g[t] + pdis[t]), f"PowerBalance_{t}"

# (8) BESS Dynamik
eta = np.sqrt(BESS_ROUNDTRIP_EFF)
for t in range(N_T):
    model += (pch[t]  <= Pbat), f"BESSchLimit_{t}"
    model += (pdis[t] <= Pbat), f"BESSdisLimit_{t}"
    model += (Eb[t+1] == Eb[t] + eta * pch[t] * DT_H - (1/eta) * pdis[t] * DT_H), f"BESSdyn_{t}"
    model += (Eb[t] <= Ebat), f"BESSmaxE_{t}"
    model += (Eb[t] >= BESS_MIN_SOC_SHARE * Ebat), f"BESSminE_{t}"
model += (Eb[N_T] == Eb[0]), "BESS_cyclic"

# (9) SOC Dynamik
for k in K:
    cap_expr = pulp.lpSum(batt_e[e] * uE[k][e] for e in E)
    for t in range(N_T):
        charge_energy = p[k][t] * DT_H
        consume_energy = pulp.lpSum(b[(r,t)] * energy_route[(r,e)] * xE[k][r][e] for r in R for e in E)
        model += (SOC[k][t+1] == SOC[k][t] + charge_energy - consume_energy), f"SOCdyn_{k}_{t}"
        model += (SOC[k][t] <= cap_expr), f"SOCcap_{k}_{t}"
    model += (SOC[k][N_T] == SOC[k][0]), f"SOC_cyclic_{k}"

# (10) Peak Definition
for t in range(N_T):
    model += (g[t] <= Gmax), f"GmaxDef_{t}"

# ----------------------------
# Zielfunktion (€/a) OHNE Tie-Break
# ----------------------------
fixed_cost = pulp.lpSum(
    uD[k] * (capex_d + opex_d + kfz_d)
    + pulp.lpSum(uE[k][e] * (capex_e[e] + opex_e[e] - thg_e[e]) for e in E)
    for k in K
)

diesel_var_cost = N_DAYS * pulp.lpSum(
    xD[k][r] * (
        TOLL_EUR_PER_KM * dist_toll[r] +              # Maut (nur Diesel)
        diesel_fuel_cost_per_km * dist_total[r]       # Spritkosten (Diesel)
    )
    for k in K for r in R
)

charger_cost = pulp.lpSum(nS[s] * (charger_capex[s] + charger_opex[s]) for s in S)
trafo_cost = TRAFO_COST_YEAR * zTrafo

bess_capex = BESS_CAPEX_PER_KW_YEAR * Pbat + BESS_CAPEX_PER_KWH_YEAR * Ebat
bess_cost  = bess_capex + BESS_OPEX_SHARE * bess_capex

energy_cost_year = N_DAYS * pulp.lpSum(g[t] * DT_H * ENERGY_PRICE_EUR_PER_KWH for t in range(N_T))
demand_cost_year = DEMAND_CHARGE_EUR_PER_KW_YEAR * Gmax

model += (
    fixed_cost + diesel_var_cost + charger_cost + trafo_cost + bess_cost
    + energy_cost_year + demand_cost_year + GRID_BASE_FEE_YEAR
)

# ----------------------------
# Solve
# ----------------------------
solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=180)
status = model.solve(solver)

print("\nStatus:", pulp.LpStatus[status])
print("Objektivwert (€/a):", pulp.value(model.objective))

# ----------------------------
# Ergebnisse
# ----------------------------
fleet_d = sum(1 for k in K if (pulp.value(uD[k]) or 0) > 0.5)
fleet_e = {e: sum(1 for k in K if (pulp.value(uE[k][e]) or 0) > 0.5) for e in E}

print("\n--- Optimale Flotte ---")
print("Diesel:", fleet_d)
for e in E:
    print(f"{e}: {fleet_e[e]}")

print("\n--- Infrastruktur ---")
print("Trafo erweitert:", int((pulp.value(zTrafo) or 0) > 0.5))
for s in S:
    print(f"{s}: {int(round(pulp.value(nS[s]) or 0))} Stück")

print("\n--- Speicher (BESS) ---")
print("Pbat (kW):", float(pulp.value(Pbat) or 0.0))
print("Ebat (kWh):", float(pulp.value(Ebat) or 0.0))

print("\n--- Netz ---")
print("Gmax (kW):", float(pulp.value(Gmax) or 0.0))

# Tourzuordnung
rows = []
for r in R:
    for k in K:
        if (pulp.value(xD[k][r]) or 0) > 0.5:
            rows.append((r, route_name[r], "Diesel", k))
        for e in E:
            if (pulp.value(xE[k][r][e]) or 0) > 0.5:
                rows.append((r, route_name[r], e, k))

assign_df = pd.DataFrame(rows, columns=["route_id","route_name","vehicle_type","vehicle_k"])
assign_df["distance_total"] = assign_df["route_id"].map(dist_total)
assign_df["distance_toll"]  = assign_df["route_id"].map(dist_toll)
assign_df = assign_df.sort_values(["vehicle_type","vehicle_k","route_id"])

print("\n--- Tourzuordnung ---")
display(assign_df)
