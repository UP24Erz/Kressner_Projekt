# ============================================================
# MAX-CLEAN MILP (PuLP/CBC) – HARD-CODED DATA (NO CSV)
# - 1 Betriebstag modelliert, Kosten auf €/Jahr hochgerechnet (N_DAYS)
# - Zeitraster fix: 15 Minuten -> 96 Zeitschritte
# - Einheiten:
#   * km, kWh, kW, €, €/a
# - OHNE Nachtregel, OHNE Tie-Break
# - OHNE Trafo, OHNE Speicher (BESS)
# - Netzlimit fix: 500 kW
# - max 5 Diesel + max 5 Elektro
# - Dieselpreis fix: 1,60 €/Liter
# ============================================================

import pulp

# ----------------------------
# 0) HARD-CODED GLOBALS
# ----------------------------
N_DAYS = 260

TIME_STEP_MIN = 15
DT_H = 0.25            # 15 min = 0.25 h
N_T = 96               # 24h / 0.25h

GRID_LIMIT_KW = 500
ENERGY_PRICE_EUR_PER_KWH = 0.25
GRID_BASE_FEE_YEAR = 1000
DEMAND_CHARGE_EUR_PER_KW_YEAR = 150  # Beispielwert aus Veranstaltung

TOLL_EUR_PER_KM = 0.34               # Diesel-Maut
DIESEL_PRICE_EUR_PER_LITER = 1.60

MAX_DIESEL = 5
MAX_ELECTRIC = 5
MAX_CHARGERS = 3

BIGM = 10_000

# ----------------------------
# 1) HARD-CODED ROUTES (alle aus deinem Screenshot)
#    times in "HH:MM"
# ----------------------------
ROUTES = {
    "t-4": {"name": "Nahverkehr",              "dist_km": 250, "toll_km": 150, "start": "06:45", "end": "17:15"},
    "t-5": {"name": "Nahverkehr",              "dist_km": 250, "toll_km": 150, "start": "06:30", "end": "17:00"},
    "t-6": {"name": "Nahverkehr",              "dist_km": 250, "toll_km": 150, "start": "06:00", "end": "16:30"},
    "s-1": {"name": "Ditzingen",               "dist_km": 120, "toll_km":  32, "start": "05:30", "end": "15:30"},
    "s-2": {"name": "Ditzingen",               "dist_km": 120, "toll_km":  32, "start": "06:00", "end": "16:00"},
    "s-3": {"name": "Ditzingen",               "dist_km": 120, "toll_km":  32, "start": "09:00", "end": "16:00"},
    "s-4": {"name": "Ditzingen",               "dist_km": 120, "toll_km":  32, "start": "06:30", "end": "16:30"},
    "w1":  {"name": "Ditzingen",               "dist_km": 100, "toll_km":  32, "start": "05:30", "end": "15:30"},
    "w2":  {"name": "Ditzingen",               "dist_km": 100, "toll_km":  32, "start": "08:00", "end": "18:00"},
    "w3":  {"name": "Ditzingen",               "dist_km": 100, "toll_km":  32, "start": "06:45", "end": "16:45"},
    "w4":  {"name": "Ditzingen",               "dist_km": 100, "toll_km":  32, "start": "06:00", "end": "16:00"},
    "w5":  {"name": "Ditzingen",               "dist_km": 100, "toll_km":  32, "start": "07:00", "end": "17:00"},
    "w6":  {"name": "Ditzingen",               "dist_km": 100, "toll_km":  32, "start": "05:30", "end": "15:30"},
    "w7":  {"name": "Ditzingen",               "dist_km": 100, "toll_km":  32, "start": "07:15", "end": "17:15"},
    "r1":  {"name": "MultiStop",               "dist_km": 285, "toll_km": 259, "start": "18:00", "end": "22:30"},
    "r2":  {"name": "MultiStop",               "dist_km": 250, "toll_km": 220, "start": "16:30", "end": "21:45"},
    "r3":  {"name": "Schramberg",              "dist_km": 235, "toll_km": 219, "start": "17:45", "end": "21:30"},
    "h3":  {"name": "Hettingen",               "dist_km": 180, "toll_km": 160, "start": "18:45", "end": "22:45"},
    "h4":  {"name": "Hettingen",               "dist_km": 180, "toll_km": 160, "start": "18:30", "end": "22:30"},
    "k1":  {"name": "Wendlingen am Neckar",    "dist_km": 275, "toll_km": 235, "start": "16:30", "end": "22:30"},
}

R = list(ROUTES.keys())

# ----------------------------
# 2) HARD-CODED VEHICLES
# ----------------------------
# Diesel (defaults – bitte ggf. mit deinen CSV-Werten überschreiben)
DIESEL = {
    "capex_yearly_eur": 24000,
    "opex_yearly_eur":  6000,
    "kfz_yearly_eur":    556,   # <- falls du z.B. 1000 hast: hier eintragen
    "avg_diesel_l_per_100km": 26.0,  # <- falls deine CSV z.B. 28 hat: hier eintragen
}

diesel_l_per_km = DIESEL["avg_diesel_l_per_100km"] / 100.0
diesel_fuel_cost_per_km = diesel_l_per_km * DIESEL_PRICE_EUR_PER_LITER

# Electric trucks (aus deinem CSV-Preview)
ELECTRIC = {
    "eActros600": {"capex_yearly_eur": 60000, "opex_yearly_eur": 6000, "thg_yearly_eur": 1000,
                   "avg_kwh_per_100km": 110, "pmax_kw": 400, "soc_max_kwh": 621},
    "eActros400": {"capex_yearly_eur": 50000, "opex_yearly_eur": 5000, "thg_yearly_eur": 1000,
                   "avg_kwh_per_100km": 105, "pmax_kw": 400, "soc_max_kwh": 414},
}
E = list(ELECTRIC.keys())

# ----------------------------
# 3) HARD-CODED CHARGERS (aus deinem CSV-Preview)
# ----------------------------
CHARGERS = {
    "Alpitronic-50":  {"capex_yearly_eur":  3000, "opex_yearly_eur": 1000, "pmax_kw":  50, "spots": 2},
    "Alpitronic-200": {"capex_yearly_eur": 10000, "opex_yearly_eur": 1500, "pmax_kw": 200, "spots": 2},
    "Alpitronic-400": {"capex_yearly_eur": 16000, "opex_yearly_eur": 2000, "pmax_kw": 400, "spots": 2},
}
S = list(CHARGERS.keys())

# ----------------------------
# 4) TIME INDEXING (hard-coded)
# ----------------------------
def hhmm_to_index(hhmm: str) -> int:
    # 15-min raster, expected exact multiples of 15 minutes in input
    hh, mm = hhmm.split(":")
    minutes = int(hh) * 60 + int(mm)
    return minutes // TIME_STEP_MIN

# a[r,t]=1 active, b[r,t]=1 ends
a = {(r, t): 0 for r in R for t in range(N_T)} #Jede Route ist zunächst nie aktiv
b = {(r, t): 0 for r in R for t in range(N_T)} # „Route r endet in Zeit t“

for r in R:
    si = hhmm_to_index(ROUTES[r]["start"])
    ei = hhmm_to_index(ROUTES[r]["end"])
    if ei < si:
        raise ValueError(f"Route {r} crosses midnight (not supported in max-clean version).")
    for t in range(si, ei):
        a[(r, t)] = 1
    b[(r, ei - 1)] = 1  # consumption booked at end slot

# Route energy (kWh) for each E-type
route_energy_kwh = {}
for r in R:
    dist = ROUTES[r]["dist_km"]
    for e in E:
        kwh_per_km = ELECTRIC[e]["avg_kwh_per_100km"] / 100.0
        route_energy_kwh[(r, e)] = dist * kwh_per_km

# ✅ Wann jede Route fährt
# ✅ Wann ein Lkw nicht laden darf
# ✅ Wann zwei Touren kollidieren
# ✅ Wann Energie abgezogen wird
# ✅ Wann ein Fahrzeug frei ist

# ----------------------------
# 5) MILP
# ----------------------------
K = list(range(len(R)))  # upper bound: one vehicle per route

m = pulp.LpProblem("MaxClean_Fleet_Charging", pulp.LpMinimize)

# --- decision variables ---
# vehicle type selection
uD = pulp.LpVariable.dicts("uD", K, 0, 1, cat="Binary")
uE = pulp.LpVariable.dicts("uE", (K, E), 0, 1, cat="Binary")

# route assignment
xD = pulp.LpVariable.dicts("xD", (K, R), 0, 1, cat="Binary")
xE = pulp.LpVariable.dicts("xE", (K, R, E), 0, 1, cat="Binary")

# chargers
nS = pulp.LpVariable.dicts("nCharger", S, lowBound=0, cat="Integer")

# charging & SOC
c = pulp.LpVariable.dicts("cCharge", (K, range(N_T)), 0, 1, cat="Binary")      # charging yes/no
p = pulp.LpVariable.dicts("pKW", (K, range(N_T)), lowBound=0, cat="Continuous") # kW
SOC = pulp.LpVariable.dicts("SOCkWh", (K, range(N_T + 1)), lowBound=0, cat="Continuous")

# grid + peak
g = pulp.LpVariable.dicts("gKW", range(N_T), lowBound=0, cat="Continuous")
Gmax = pulp.LpVariable("Gmax", lowBound=0, cat="Continuous")

def driving_expr(k, t):
    return (pulp.lpSum(a[(r, t)] * xD[k][r] for r in R)
            + pulp.lpSum(a[(r, t)] * xE[k][r][e] for r in R for e in E))

# ----------------------------
# 6) CONSTRAINTS
# ----------------------------

# fleet limits, Es dürfen maximal 5 Diesel- und 5 Elektro-Lkw eingesetzt werden.
m += pulp.lpSum(uD[k] for k in K) <= MAX_DIESEL
m += pulp.lpSum(uE[k][e] for k in K for e in E) <= MAX_ELECTRIC

# each route exactly once
for r in R:
    m += pulp.lpSum(xD[k][r] for k in K) + pulp.lpSum(xE[k][r][e] for k in K for e in E) == 1

# per vehicle: Ein Fahrzeug ist entweder Diesel oder Elektro – nicht beides.
for k in K:
    m += uD[k] + pulp.lpSum(uE[k][e] for e in E) <= 1
    for r in R:
        m += xD[k][r] <= uD[k]
        for e in E:
            m += xE[k][r][e] <= uE[k][e]

# no parallel routes per vehicle
for k in K:
    for t in range(N_T):
        m += driving_expr(k, t) <= 1

# charging only if electric & not driving
for k in K:
    eBought = pulp.lpSum(uE[k][e] for e in E)
    for t in range(N_T):
        m += c[k][t] <= eBought
        m += c[k][t] + driving_expr(k, t) <= 1
        m += p[k][t] <= BIGM * c[k][t]
        m += p[k][t] <= pulp.lpSum(ELECTRIC[e]["pmax_kw"] * uE[k][e] for e in E)

# Es dürfen höchstens 3 MAX_CHARGERS Ladesäulen
m += pulp.lpSum(nS[s] for s in S) <= MAX_CHARGERS

# Zu jedem Zeitpunkt dürfen nicht mehr Lkw laden, als Ladepunkte vorhanden sind.
for t in range(N_T):
    m += pulp.lpSum(c[k][t] for k in K) <= pulp.lpSum(CHARGERS[s]["spots"] * nS[s] for s in S)
    m += pulp.lpSum(p[k][t] for k in K) <= pulp.lpSum(CHARGERS[s]["pmax_kw"] * nS[s] for s in S)

# „Der Netzanschluss begrenzt die Leistung, und der Peak kostet Geld.“
for t in range(N_T):
    m += g[t] <= GRID_LIMIT_KW
    m += pulp.lpSum(p[k][t] for k in K) <= g[t]
    m += g[t] <= Gmax

# SOC dynamics (consumption booked at route end)
for k in K:
    cap = pulp.lpSum(ELECTRIC[e]["soc_max_kwh"] * uE[k][e] for e in E)
    for t in range(N_T):
        charge_kwh = p[k][t] * DT_H
        consume_kwh = pulp.lpSum(b[(r, t)] * route_energy_kwh[(r, e)] * xE[k][r][e] for r in R for e in E)
        m += SOC[k][t + 1] == SOC[k][t] + charge_kwh - consume_kwh
        m += SOC[k][t] <= cap
    m += SOC[k][N_T] == SOC[k][0]  # day cycle

# ----------------------------
# 7) OBJECTIVE (€/year)
# ----------------------------
fleet_cost = pulp.lpSum(
    uD[k] * (DIESEL["capex_yearly_eur"] + DIESEL["opex_yearly_eur"] + DIESEL["kfz_yearly_eur"])
    + pulp.lpSum(uE[k][e] * (ELECTRIC[e]["capex_yearly_eur"] + ELECTRIC[e]["opex_yearly_eur"] - ELECTRIC[e]["thg_yearly_eur"])
                 for e in E)
    for k in K
)

diesel_var_cost = N_DAYS * pulp.lpSum(
    xD[k][r] * (TOLL_EUR_PER_KM * ROUTES[r]["toll_km"] + diesel_fuel_cost_per_km * ROUTES[r]["dist_km"])
    for k in K for r in R
)

charger_cost = pulp.lpSum(
    nS[s] * (CHARGERS[s]["capex_yearly_eur"] + CHARGERS[s]["opex_yearly_eur"])
    for s in S
)

energy_cost = N_DAYS * pulp.lpSum(g[t] * DT_H * ENERGY_PRICE_EUR_PER_KWH for t in range(N_T))
peak_cost = DEMAND_CHARGE_EUR_PER_KW_YEAR * Gmax

m += fleet_cost + diesel_var_cost + charger_cost + energy_cost + peak_cost + GRID_BASE_FEE_YEAR

# ----------------------------
# 8) SOLVE & PRINT
# ----------------------------
m.solve(pulp.PULP_CBC_CMD(msg=True, timeLimit=180))

print("\nStatus:", pulp.LpStatus[m.status])
print("Objective (€/a):", pulp.value(m.objective))

diesel_cnt = sum(1 for k in K if (pulp.value(uD[k]) or 0) > 0.5)
e_cnt = {e: sum(1 for k in K if (pulp.value(uE[k][e]) or 0) > 0.5) for e in E}

print("\n--- Fleet ---")
print(f"Diesel: {diesel_cnt} (max {MAX_DIESEL})")
print(f"Electric total: {sum(e_cnt.values())} (max {MAX_ELECTRIC})")
for e in E:
    print(f"{e}: {e_cnt[e]}")

print("\n--- Chargers ---")
for s in S:
    print(f"{s}: {int(round(pulp.value(nS[s]) or 0))}")

print("\n--- Grid ---")
print("Gmax (kW):", float(pulp.value(Gmax) or 0.0))

print("\n--- Assignments (route -> vehicle) ---")
for r in R:
    assigned = None
    for k in K:
        if (pulp.value(xD[k][r]) or 0) > 0.5:
            assigned = f"Diesel (k={k})"
        for e in E:
            if (pulp.value(xE[k][r][e]) or 0) > 0.5:
                assigned = f"{e} (k={k})"
    print(f"{r:>3} | {ROUTES[r]['name']:<22} {ROUTES[r]['start']}-{ROUTES[r]['end']} -> {assigned}")
