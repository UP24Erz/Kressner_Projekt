r"""
================================================================================
MILP-Modell zur Optimierung der Elektrifizierung eines Logistikdepots
================================================================================

Mathematische Formulierung:
---------------------------

INDEXMENGEN:
-----------
V_E    : Menge aller e-Lkw-Instanzen (indexiert: v ∈ V_E)
V_D    : Menge aller Diesel-Lkw (indexiert: v ∈ V_D)
V      : V_E ∪ V_D (alle Fahrzeuge)
M_E    : Menge aller e-Lkw-Typen/Modelle (z.B. eActros400, eActros600)
M_D    : Menge aller Diesel-Lkw-Typen
R      : Menge aller Touren (indexiert: r ∈ R)
T      : Menge aller Zeitschritte (t ∈ {1,...,96}, je 15min)
C      : Menge aller Ladesäulentypen (indexiert: c ∈ C)
S      : Menge aller Ladepunkte (indexiert: s ∈ S)

PARAMETER:
----------
Zeitliche Parameter:
  Δt                : Zeitschrittlänge [h] (0.25h = 15min)
  N_days            : Anzahl Betriebstage pro Jahr [d] (260)
  
Tourenparameter:
  d_r               : Gesamtdistanz von Tour r [km]
  d_r^toll          : Mautpflichtige Distanz von Tour r [km]
  t_r^start         : Startzeit von Tour r [Zeitschritt]
  t_r^end           : Endzeit von Tour r [Zeitschritt]
  
E-Lkw Parameter:
  CAPEX_m^E         : Jährliche Leasingkosten für e-Lkw-Modell m ∈ M_E [€/a]
  OPEX_m^E          : Jährliche Wartungskosten für e-Lkw-Modell m [€/a]
  ε_m               : Energieverbrauch von e-Lkw-Modell m [kWh/km]
  Q_m               : Batteriekapazität von e-Lkw-Modell m [kWh]
  P_m^charge,max    : Max. Ladeleistung von e-Lkw-Modell m [kW]
  THG_m             : THG-Quotenerlös für e-Lkw-Modell m [€/a]
  
Diesel-Lkw Parameter:
  CAPEX_m^D         : Jährliche Leasingkosten für Diesel-Lkw-Modell m ∈ M_D [€/a]
  OPEX_m^D          : Jährliche Wartungskosten für Diesel-Lkw-Modell m [€/a]
  TAX_m^D           : Jährliche KFZ-Steuer für Diesel-Lkw-Modell m [€/a]
  κ_m               : Dieselverbrauch von Diesel-Lkw-Modell m [L/km]
  
Ladeinfrastruktur Parameter:
  CAPEX_c^charge    : Jährliche Investitionskosten für Ladesäule Typ c [€/a]
  OPEX_c^charge     : Jährliche Betriebskosten für Ladesäule Typ c [€/a]
  P_c^max           : Maximale Leistung von Ladesäule Typ c [kW]
  n_c^spots         : Anzahl Ladepunkte an Ladesäule Typ c [-]
  N_c^max           : Maximale Anzahl installierbarer Ladesäulen Typ c [-]
  
Batteriespeicher Parameter:
  CAPEX_stor^P      : Spezifische Investitionskosten Speicherleistung [€/(kW·a)]
  CAPEX_stor^E      : Spezifische Investitionskosten Speicherkapazität [€/(kWh·a)]
  OPEX_stor^P       : Spezifische Betriebskosten Speicherleistung [€/(kW·a)]
  OPEX_stor^E       : Spezifische Betriebskosten Speicherkapazität [€/(kWh·a)]
  η_charge          : Ladewirkungsgrad Speicher [-] (0.98)
  η_discharge       : Entladewirkungsgrad Speicher [-] (0.98)
  DoD_min           : Minimale Entladetiefe Speicher [-] (0.025)
  
Netzanschluss Parameter:
  P_grid^base       : Basis-Netzanschlussleistung [kW] (500)
  P_grid^upgrade    : Erweiterung Netzanschlussleistung [kW] (500)
  COST_grid^upgrade : Jährliche Kosten für Netzerweiterung [€/a]
  COST_grid^base    : Jährliche Grundgebühr Netzanschluss [€/a]
  
Energiekosten:
  c_el              : Arbeitspreis Strom [€/kWh]
  c_peak            : Leistungspreis Strom [€/(kW·a)]
  c_diesel          : Dieselpreis [€/L]
  c_toll            : Mautpreis [€/km]
  

ENTSCHEIDUNGSVARIABLEN:
-----------------------
Binäre Variablen:
  x_v               : 1, falls e-Lkw/Diesel-Lkw v beschafft wird, 0 sonst
  y_{v,r}           : 1, falls Fahrzeug v Tour r zugeordnet wird, 0 sonst
  z_c               : Anzahl installierter Ladesäulen vom Typ c (ganzzahlig)
  w_{v,s,t}         : 1, falls Fahrzeug v zur Zeit t an Ladepunkt s lädt, 0 sonst
  u_grid            : 1, falls Netzerweiterung durchgeführt wird, 0 sonst
  
Kontinuierliche Variablen:
  SOC_{v,t}         : State of Charge von e-Lkw v zur Zeit t [kWh]
  p_{v,s,t}         : Ladeleistung von e-Lkw v an Ladepunkt s zur Zeit t [kW]
  p_t^grid          : Netzbezugsleistung zur Zeit t [kW]
  p^peak            : Jahreshöchstlast Netzbezug [kW]
  P^stor            : Installierte Speicherleistung [kW]
  E^stor            : Installierte Speicherkapazität [kWh]
  p_t^stor,charge   : Speicher-Ladeleistung zur Zeit t [kW]
  p_t^stor,discharge: Speicher-Entladeleistung zur Zeit t [kW]
  SOC_t^stor        : State of Charge Speicher zur Zeit t [kWh]

ZIELFUNKTION:
-------------
Minimiere Gesamtkosten (Total Cost of Ownership):

min Z = Z_vehicles^E + Z_vehicles^D + Z_infrastructure + Z_grid + Z_storage + Z_energy - Z_revenue

wobei:

Z_vehicles^E      = ∑_{v∈V_E} x_v · (CAPEX_{model(v)}^E + OPEX_{model(v)}^E)
                    [Jährliche Kosten e-Lkw-Flotte]
                    
Z_vehicles^D      = ∑_{v∈V_D} x_v · (CAPEX_{model(v)}^D + OPEX_{model(v)}^D + TAX_{model(v)}^D)
                    [Jährliche Kosten Diesel-Flotte]
                    
Z_infrastructure  = ∑_{c∈C} z_c · (CAPEX_c^charge + OPEX_c^charge)
                    [Jährliche Kosten Ladeinfrastruktur]
                    
Z_grid            = COST_grid^base + u_grid · COST_grid^upgrade
                    [Jährliche Netzanschlusskosten]
                    
Z_storage         = P^stor · (CAPEX_stor^P + OPEX_stor^P) + 
                    E^stor · (CAPEX_stor^E + OPEX_stor^E)
                    [Jährliche Speicherkosten]
                    
Z_energy          = Z_electricity + Z_diesel + Z_toll
                    
  Z_electricity   = N_days · (c_el · ∑_{t∈T} p_t^grid · Δt + c_peak · p^peak)
                    [Jährliche Stromkosten: Arbeit + Leistung]
                    
  Z_diesel        = N_days · c_diesel · ∑_{v∈V_D} ∑_{r∈R} y_{v,r} · κ_{model(v)} · d_r
                    [Jährliche Dieselkosten]
                    
  Z_toll          = N_days · c_toll · ∑_{v∈V_D} ∑_{r∈R} y_{v,r} · d_r^toll
                    [Jährliche Mautkosten (nur Diesel)]
                    
Z_revenue         = ∑_{v∈V_E} x_v · THG_{model(v)}
                    [THG-Quotenerlöse e-Lkw]

NEBENBEDINGUNGEN:
-----------------

(1) Tourenzuordnung - Jede Tour muss genau einmal gefahren werden:
    ∑_{v∈V} y_{v,r} = 1                                    ∀r ∈ R

(2) Fahrzeugexistenz - Touren nur mit beschafften Fahrzeugen:
    y_{v,r} ≤ x_v                                          ∀v ∈ V, ∀r ∈ R

(3) Keine zeitliche Überlappung - Ein Fahrzeug kann nicht gleichzeitig zwei Touren fahren:
    ∑_{r∈R: t∈[t_r^start,t_r^end]} y_{v,r} ≤ 1           ∀v ∈ V, ∀t ∈ T

(4) Ladeinfrastruktur - Maximal 3 Ladesäulen INSGESAMT:
    ∑_{c∈C} z_c ≤ 3

(5) Ladepunkt-Belegungs-Kapazität - Anzahl gleichzeitig ladender Lkw begrenzt:
    ∑_{v∈V_E} ∑_{s∈S_c} w_{v,s,t} ≤ n_c^spots · z_c      ∀c ∈ C, ∀t ∈ T

(6) Keine gleichzeitige Fahrt und Ladung:
    ∑_{r∈R: t∈[t_r^start,t_r^end]} y_{v,r} + ∑_{s∈S} w_{v,s,t} ≤ 1    ∀v ∈ V_E, ∀t ∈ T

(7) Ladeleistung nur bei Belegung (Linearisierung):
    p_{v,s,t} ≤ M · w_{v,s,t}                             ∀v ∈ V_E, ∀s ∈ S, ∀t ∈ T

(8) Maximale Fahrzeugladeleistung:(wird nicht mehr verwendet)
    p_{v,s,t} ≤ P_{model(v)}^charge,max · w_{v,s,t}      ∀v ∈ V_E, ∀s ∈ S, ∀t ∈ T

(9) Ladesäulen-Leistungsgrenze - Summe über alle Ladepunkte einer Säule:
    ∑_{v∈V_E} ∑_{s∈S_c} p_{v,s,t} ≤ P_c^max · z_c        ∀c ∈ C, ∀t ∈ T

(10) Operative Depot-Regeln - Nacht-Ladekontinuität (18:00-6:00):
     Vereinfachte Regel: Lkw, die nachts im Depot sind und laden, tun dies kontinuierlich

(11) Batteriebilanz - SOC-Dynamik:
     SOC_{v,t+1} = SOC_{v,t} + Δt · (∑_{s∈S} p_{v,s,t} - ε_{model(v)} · ∑_{r∈R: t∈[t_r^start,t_r^end]} y_{v,r} · d_r/(t_r^end - t_r^start + 1)/Δt)
                                                           ∀v ∈ V_E, ∀t ∈ T\{|T|}

(12) Zyklischer SOC - Ende = Anfang:
     SOC_{v,1} = SOC_{v,|T|}                              ∀v ∈ V_E

(13) SOC-Grenzen:
     0 ≤ SOC_{v,t} ≤ Q_{model(v)} · x_v                   ∀v ∈ V_E, ∀t ∈ T

(14) Leistungsbilanz - Grid + Speicher-Entladung = Laden + Speicher-Ladung:
     p_t^grid + p_t^stor,discharge = ∑_{v∈V_E} ∑_{s∈S} p_{v,s,t} + p_t^stor,charge    ∀t ∈ T

(15) Netzanschlussgrenze:
     p_t^grid ≤ P_grid^base + u_grid · P_grid^upgrade     ∀t ∈ T

(16) Peak-Definition - Jahreshöchstlast:
     p^peak ≥ p_t^grid                                     ∀t ∈ T

(17) Speicher-Leistungsgrenze:
     p_t^stor,charge ≤ P^stor                             ∀t ∈ T
     p_t^stor,discharge ≤ P^stor                          ∀t ∈ T

(18) Speicher-Bilanz:
     SOC_t+1^stor = SOC_t^stor + Δt · (η_charge · p_t^stor,charge - p_t^stor,discharge/η_discharge)
                                                           ∀t ∈ T\{|T|}

(19) Zyklischer Speicher-SOC:
     SOC_1^stor = SOC_|T|^stor

(20) Speicher-SOC-Grenzen mit Mindest-DoD:
     DoD_min · E^stor ≤ SOC_t^stor ≤ E^stor               ∀t ∈ T

(21) Nicht-Negativität:
     Alle kontinuierlichen Variablen ≥ 0

================================================================================
"""

from pyscipopt import Model, quicksum, SCIP_PARAMSETTING
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ============================================================================
# KONFIGURATION UND PARAMETER
# ============================================================================

class OptimizationConfig:
    """Zentrale Konfigurationsklasse für alle Modellparameter"""
    
    # Zeitliche Parameter
    DELTA_T = 0.25              # Zeitschrittlänge [h] (15 Minuten)
    NUM_TIMESTEPS = 96          # Anzahl Zeitschritte pro Tag
    NUM_DAYS = 260              # Betriebstage pro Jahr
    
    # Energiepreise
    ELECTRICITY_PRICE = 0.25    # [€/kWh] Arbeitspreis Strom
    PEAK_PRICE = 150.0          # [€/(kW·a)] Leistungspreis
    DIESEL_PRICE = 1.50         # [€/L] Dieselpreis
    TOLL_PRICE = 0.34           # [€/km] Mautpreis
    
    # Netzanschluss
    GRID_BASE_POWER = 500.0     # [kW] Basis-Anschlussleistung
    GRID_UPGRADE_POWER = 500.0  # [kW] Erweiterung
    GRID_UPGRADE_COST = 10000.0 # [€/a] Kosten Erweiterung
    GRID_BASE_FEE = 1000.0      # [€/a] Grundgebühr
    
    # Batteriespeicher
    STORAGE_CAPEX_POWER = 30.0     # [€/(kW·a)] Investition Leistung
    STORAGE_CAPEX_ENERGY = 350.0    # [€/(kWh·a)] Investition Kapazität
    STORAGE_OPEX_POWER = 0.02 * STORAGE_CAPEX_POWER        # [€/(kW·a)] Betrieb Leistung
    STORAGE_OPEX_ENERGY = 0.02 * STORAGE_CAPEX_ENERGY       # [€/(kWh·a)] Betrieb Kapazität
    STORAGE_ETA_CHARGE = 0.98       # [-] Ladewirkungsgrad
    STORAGE_ETA_DISCHARGE = 0.98    # [-] Entladewirkungsgrad
    STORAGE_DOD_MIN = 0.025         # [-] Minimale Entladetiefe (2.5%)
    
    # Ladeinfrastruktur
    MAX_CHARGERS_PER_TYPE = 3   # Maximal 3 Ladesäulen insgesamt
    
    # Solver-Einstellungen
    SOLVER_TIME_LIMIT = 29000    # [s] ca. 8 Stunden
    SOLVER_GAP = 0.02          # 2% Optimalitätslücke
    
    # Big-M für Linearisierung
    BIG_M = 10000.0


# ============================================================================
# DATEN-IMPORT
# ============================================================================

def load_data(data_dir="Daten"):
    """
    Lädt alle CSV-Dateien und bereitet Datenstrukturen vor
    
    Returns:
        dict mit Schlüsseln: 'routes', 'electric_trucks', 'diesel_trucks', 'chargers'
    """
    data = {}
    
    # Routen einlesen
    routes_path = os.path.join(data_dir, "routes.csv")
    data['routes'] = pd.read_csv(routes_path, sep=';', decimal=',')
    print(f"✓ {len(data['routes'])} Routen geladen")
    
    # E-Lkw einlesen
    etruck_path = os.path.join(data_dir, "electric_trucks.csv")
    data['electric_trucks'] = pd.read_csv(etruck_path, sep=';', decimal=',')
    print(f"✓ {len(data['electric_trucks'])} E-Lkw-Modelle geladen")
    
    # Diesel-Lkw einlesen
    dtruck_path = os.path.join(data_dir, "diesel_trucks.csv")
    data['diesel_trucks'] = pd.read_csv(dtruck_path, sep=';', decimal=',')
    print(f"✓ {len(data['diesel_trucks'])} Diesel-Lkw-Modelle geladen")
    
    # Ladesäulen einlesen
    chargers_path = os.path.join(data_dir, "chargers.csv")
    data['chargers'] = pd.read_csv(chargers_path, sep=';', decimal=',')
    print(f"✓ {len(data['chargers'])} Ladesäulentypen geladen")
    
    return data


def parse_time_to_timestep(time_str, delta_t=0.25):
    """
    Konvertiert Zeitstring (HH:MM) in Zeitschritt-Index (0-95)
    
    Args:
        time_str: Zeit als String, z.B. "06:45"
        delta_t: Zeitschrittlänge in Stunden (0.25 für 15min)
    
    Returns:
        int: Zeitschritt-Index (0-basiert)
    """
    h, m = map(int, time_str.split(':'))
    hours_since_midnight = h + m/60.0
    timestep = int(hours_since_midnight / delta_t)
    return timestep


# ============================================================================
# INDEXMENGEN AUFBAU
# ============================================================================

def build_index_sets(data):
    """
    Erstellt alle Indexmengen basierend auf den eingelesenen Daten
    
    Returns:
        dict mit allen Indexmengen
    """
    sets = {}
    
    # R: Menge aller Touren
    sets['R'] = list(data['routes']['route_id'])
    num_routes = len(sets['R'])
    print(f"\n│ R │ = {num_routes} Touren")
    
    # M_E: Menge aller E-Lkw-Modelle
    sets['M_E'] = list(data['electric_trucks']['truck_model'])
    print(f"│M_E│ = {len(sets['M_E'])} E-Lkw-Typen: {sets['M_E']}")
    
    # M_D: Menge aller Diesel-Lkw-Modelle
    sets['M_D'] = list(data['diesel_trucks']['truck_model'])
    print(f"│M_D│ = {len(sets['M_D'])} Diesel-Typen: {sets['M_D']}")
    
    # V_E: Menge aller E-Lkw-Instanzen (eine pro Route pro Modell - pessimistisch)
    # Wir erstellen potenzielle Instanzen: für jedes Modell genug für alle Routen
    sets['V_E'] = []
    sets['model_map_E'] = {}  # Mapping: Instanz -> Modell
    for model in sets['M_E']:
        for i in range(num_routes):  # Limit auf 4 pro Modell für Skalierbarkeit
            instance_id = f"e_{model}_{i}"
            sets['V_E'].append(instance_id)
            sets['model_map_E'][instance_id] = model
    print(f"│V_E│ = {len(sets['V_E'])} E-Lkw-Instanzen (Max-Bedarf)")
    
    # V_D: Menge aller Diesel-Lkw-Instanzen
    sets['V_D'] = []
    sets['model_map_D'] = {}
    for model in sets['M_D']:
        for i in range(num_routes):  # Limit auf 16 pro Modell für Skalierbarkeit
            instance_id = f"d_{model}_{i}"
            sets['V_D'].append(instance_id)
            sets['model_map_D'][instance_id] = model
    print(f"│V_D│ = {len(sets['V_D'])} Diesel-Lkw-Instanzen (Max-Bedarf)")
    
    # V: Alle Fahrzeuge
    sets['V'] = sets['V_E'] + sets['V_D']
    print(f"│ V │ = {len(sets['V'])} Fahrzeuge gesamt")
    
    # T: Zeitschritte (0 bis 95)
    sets['T'] = list(range(OptimizationConfig.NUM_TIMESTEPS))
    print(f"│ T │ = {len(sets['T'])} Zeitschritte à {OptimizationConfig.DELTA_T}h")
    
    # C: Ladesäulentypen
    sets['C'] = list(data['chargers']['charger_model'])
    print(f"│ C │ = {len(sets['C'])} Ladesäulentypen: {sets['C']}")
    print("│ S │ = Ladepunkte AGGREGIERT (keine explizite Modellierung mehr)")
    
    return sets


# ============================================================================
# PARAMETER EXTRAKTION
# ============================================================================

def extract_parameters(data, sets):
    """
    Extrahiert alle Parameter aus den Daten
    
    Returns:
        dict mit allen Parametern
    """
    params = {}
    
    # -------------------------
    # Touren-Parameter
    # -------------------------
    params['distance_total'] = {}      # d_r [km]
    params['distance_toll'] = {}       # d_r^toll [km]
    params['tour_start'] = {}          # t_r^start [Zeitschritt]
    params['tour_end'] = {}            # t_r^end [Zeitschritt]
    
    for _, row in data['routes'].iterrows():
        r = row['route_id']
        params['distance_total'][r] = row['distance_total']
        params['distance_toll'][r] = row['distance_toll']
        params['tour_start'][r] = parse_time_to_timestep(row['starttime'])
        params['tour_end'][r] = parse_time_to_timestep(row['endtime'])
    
    print(f"\n✓ Touren-Parameter extrahiert ({len(params['distance_total'])} Routen)")
    
    # -------------------------
    # E-Lkw-Parameter
    # -------------------------
    params['e_capex'] = {}           # CAPEX_m^E [€/a]
    params['e_opex'] = {}            # OPEX_m^E [€/a]
    params['e_energy_cons'] = {}     # ε_m [kWh/km]
    params['e_battery_cap'] = {}     # Q_m [kWh]
    params['e_charge_power'] = {}    # P_m^charge,max [kW]
    params['e_thg_revenue'] = {}     # THG_m [€/a]
    
    for _, row in data['electric_trucks'].iterrows():
        m = row['truck_model']
        params['e_capex'][m] = row['capex_yearly']
        params['e_opex'][m] = row['opex_yearly']
        params['e_energy_cons'][m] = row['avg_energy_kWh_per_100km'] / 100.0
        params['e_battery_cap'][m] = row['soc_max_kWh']
        params['e_charge_power'][m] = row['max_power']
        params['e_thg_revenue'][m] = row['thg_yearly']
    
    print(f"✓ E-Lkw-Parameter extrahiert ({len(params['e_capex'])} Modelle)")
    
    # -------------------------
    # Diesel-Lkw-Parameter
    # -------------------------
    params['d_capex'] = {}           # CAPEX_m^D [€/a]
    params['d_opex'] = {}            # OPEX_m^D [€/a]
    params['d_tax'] = {}             # TAX_m^D [€/a]
    params['d_fuel_cons'] = {}       # κ_m [L/km]
    
    for _, row in data['diesel_trucks'].iterrows():
        m = row['truck_model']
        params['d_capex'][m] = row['capex_yearly']
        params['d_opex'][m] = row['opex_yearly']
        params['d_tax'][m] = row['kfz_yearly']
        params['d_fuel_cons'][m] = row['avg_diesel_per_100km'] / 100.0
    
    print(f"✓ Diesel-Lkw-Parameter extrahiert ({len(params['d_capex'])} Modelle)")
    
    # -------------------------
    # Ladesäulen-Parameter
    # -------------------------
    params['charger_capex'] = {}     # CAPEX_c^charge [€/a]
    params['charger_opex'] = {}      # OPEX_c^charge [€/a]
    params['charger_power'] = {}     # P_c^max [kW]
    params['charger_spots'] = {}     # n_c^spots [-]
    
    for _, row in data['chargers'].iterrows():
        c = row['charger_model']
        params['charger_capex'][c] = row['capex_yearly']
        params['charger_opex'][c] = row['opex_yearly']
        params['charger_power'][c] = row['max_power']
        params['charger_spots'][c] = int(row['charging_spots'])
    
    print(f"✓ Ladesäulen-Parameter extrahiert ({len(params['charger_capex'])} Typen)")
    
    return params


# ============================================================================
# MODELL-AUFBAU
# ============================================================================

def build_model(data, sets, params):
    """
    Erstellt das vollständige MILP-Modell
    
    Returns:
        Model: PySCIPOpt-Modell-Objekt
    """
    print("\n" + "="*80)
    print("MODELLAUFBAU STARTET")
    print("="*80)
    
    model = Model("Depot_Electrification_MILP")
    
    # Solver-Einstellungen
    model.setRealParam('limits/time', OptimizationConfig.SOLVER_TIME_LIMIT)
    model.setRealParam('limits/gap', OptimizationConfig.SOLVER_GAP)
    model.setPresolve(SCIP_PARAMSETTING.AGGRESSIVE)
    model.setIntParam('presolving/maxrounds', -1)  # Unbegrenzt
    model.setBoolParam('constraints/linear/presolpairwise', True)
    
    # ========================================================================
    # VARIABLEN-DEKLARATION
    # ========================================================================
    print("\n[1/6] Variablen werden erstellt...")
    
    # (1) Binäre Fahrzeugbeschaffung: x_v
    x = {}
    for v in sets['V']:
        x[v] = model.addVar(vtype="BINARY", name=f"x_{v}")
    print(f"    ✓ {len(x)} Beschaffungsvariablen x_v")
    
    # (2) Binäre Tourenzuordnung: y_{v,r}
    y = {}
    for v in sets['V']:
        for r in sets['R']:
            y[v, r] = model.addVar(vtype="BINARY", name=f"y_{v}_{r}")
    print(f"    ✓ {len(y)} Zuordnungsvariablen y_{{v,r}}")
    
    # (3) Ganzzahlige Ladesäulen: z_c
    z = {}
    for c in sets['C']:
        z[c] = model.addVar(vtype="INTEGER", lb=0, ub=OptimizationConfig.MAX_CHARGERS_PER_TYPE, 
                            name=f"z_{c}")
    print(f"    ✓ {len(z)} Ladesäulen-Variablen z_c")
    
    # (4) Binäre Ladeentscheidung: w_{v,t} (AGGREGIERT ohne Spot-Zuordnung)
    w = {}
    for v in sets['V_E']:
        for t in sets['T']:
            w[v, t] = model.addVar(vtype="BINARY", name=f"w_{v}_{t}")
    print(f"    ✓ {len(w)} Lade-Belegungsvariablen w_{{v,t}} (AGGREGIERT)")
    
    # (5) Kontinuierliche Ladeleistung: p_{v,t} (AGGREGIERT ohne Spot-Zuordnung)
    p_charge = {}
    for v in sets['V_E']:
        for t in sets['T']:
            p_charge[v, t] = model.addVar(vtype="CONTINUOUS", lb=0, 
                                           name=f"p_{v}_{t}")
    print(f"    ✓ {len(p_charge)} Ladeleistungs-Variablen p_{{v,t}} (AGGREGIERT)")
    
    # (5a) Binäre Typ-Zuordnung: a_{v,t,c} - Truck v hängt zur Zeit t an Säulentyp c
    a = {}
    for v in sets['V_E']:
        for t in sets['T']:
            for c in sets['C']:
                a[v, t, c] = model.addVar(vtype="BINARY", name=f"a_{v}_{t}_{c}")
    print(f"    ✓ {len(a)} Typ-Zuordnungs-Variablen a_{{v,t,c}}")
    
    # (5b) Kontinuierliche typspezifische Ladeleistung: p_by_{v,t,c}
    p_by = {}
    for v in sets['V_E']:
        for t in sets['T']:
            for c in sets['C']:
                p_by[v, t, c] = model.addVar(vtype="CONTINUOUS", lb=0, 
                                              name=f"p_by_{v}_{t}_{c}")
    print(f"    ✓ {len(p_by)} Typspezifische Ladeleistungs-Variablen p_by_{{v,t,c}}")

    # (6) SOC für E-Lkw: SOC_{v,t}
    soc = {}
    for v in sets['V_E']:
        for t in sets['T']:
            soc[v, t] = model.addVar(vtype="CONTINUOUS", lb=0, name=f"SOC_{v}_{t}")
    print(f"    ✓ {len(soc)} SOC-Variablen SOC_{{v,t}}")
    
    # (7) Netzbezugsleistung: p_t^grid
    p_grid = {}
    for t in sets['T']:
        p_grid[t] = model.addVar(vtype="CONTINUOUS", lb=0, name=f"p_grid_{t}")
    print(f"    ✓ {len(p_grid)} Netzbezugs-Variablen p_t^grid")
    
    # (8) Peak-Leistung: p^peak
    p_peak = model.addVar(vtype="CONTINUOUS", lb=0, name="p_peak")
    print(f"    ✓ 1 Peak-Variable p^peak")
    
    # (9) Binäre Netzerweiterung: u_grid
    u_grid = model.addVar(vtype="BINARY", name="u_grid")
    print(f"    ✓ 1 Netzerweiterungs-Variable u_grid")
    
    # (10) Batteriespeicher: P^stor, E^stor
    P_stor = model.addVar(vtype="CONTINUOUS", lb=0, name="P_stor")
    E_stor = model.addVar(vtype="CONTINUOUS", lb=0, name="E_stor")
    print(f"    ✓ 2 Speicher-Dimensionierungs-Variablen P^stor, E^stor")
    
    # (11) Speicher-Leistung: p_t^stor,charge, p_t^stor,discharge
    p_stor_charge = {}
    p_stor_discharge = {}
    for t in sets['T']:
        p_stor_charge[t] = model.addVar(vtype="CONTINUOUS", lb=0, 
                                         name=f"p_stor_charge_{t}")
        p_stor_discharge[t] = model.addVar(vtype="CONTINUOUS", lb=0, 
                                            name=f"p_stor_discharge_{t}")
    print(f"    ✓ {len(p_stor_charge) + len(p_stor_discharge)} Speicher-Leistungs-Variablen")
    
    # (12) Speicher-SOC: SOC_t^stor
    soc_stor = {}
    for t in sets['T']:
        soc_stor[t] = model.addVar(vtype="CONTINUOUS", lb=0, name=f"SOC_stor_{t}")
    print(f"    ✓ {len(soc_stor)} Speicher-SOC-Variablen SOC_t^stor")
    
    # ========================================================================
    # ZIELFUNKTION
    # ========================================================================
    print("\n[2/6] Zielfunktion wird aufgebaut...")
    
    # Z_vehicles^E: E-Lkw Kosten
    z_vehicles_e = quicksum(
        x[v] * (params['e_capex'][sets['model_map_E'][v]] + 
                params['e_opex'][sets['model_map_E'][v]])
        for v in sets['V_E']
    )
    print("    ✓ Z_vehicles^E: E-Lkw-Kosten")
    
    # Z_vehicles^D: Diesel-Lkw Kosten
    z_vehicles_d = quicksum(
        x[v] * (params['d_capex'][sets['model_map_D'][v]] + 
                params['d_opex'][sets['model_map_D'][v]] +
                params['d_tax'][sets['model_map_D'][v]])
        for v in sets['V_D']
    )
    print("    ✓ Z_vehicles^D: Diesel-Lkw-Kosten")
    
    # Z_infrastructure: Ladeinfrastruktur
    z_infrastructure = quicksum(
        z[c] * (params['charger_capex'][c] + params['charger_opex'][c])
        for c in sets['C']
    )
    print("    ✓ Z_infrastructure: Ladeinfrastruktur-Kosten")
    
    # Z_grid: Netzanschluss
    z_grid = (OptimizationConfig.GRID_BASE_FEE + 
              u_grid * OptimizationConfig.GRID_UPGRADE_COST)
    print("    ✓ Z_grid: Netzanschluss-Kosten")
    
    # Z_storage: Batteriespeicher
    z_storage = (P_stor * (OptimizationConfig.STORAGE_CAPEX_POWER + 
                           OptimizationConfig.STORAGE_OPEX_POWER) +
                 E_stor * (OptimizationConfig.STORAGE_CAPEX_ENERGY + 
                           OptimizationConfig.STORAGE_OPEX_ENERGY))
    print("    ✓ Z_storage: Speicher-Kosten")
    
    # Z_electricity: Stromkosten (Arbeit + Leistung)
    z_electricity_work = OptimizationConfig.NUM_DAYS * OptimizationConfig.ELECTRICITY_PRICE * quicksum(
        p_grid[t] * OptimizationConfig.DELTA_T for t in sets['T']
    )
    z_electricity_peak =  OptimizationConfig.PEAK_PRICE * p_peak
    z_electricity = z_electricity_work + z_electricity_peak
    print("    ✓ Z_electricity: Strom-Kosten (Arbeit + Leistung)")
    
    # Z_diesel: Dieselkosten
    z_diesel = OptimizationConfig.NUM_DAYS * OptimizationConfig.DIESEL_PRICE * quicksum(
        y[v, r] * params['d_fuel_cons'][sets['model_map_D'][v]] * params['distance_total'][r]
        for v in sets['V_D'] for r in sets['R']
    )
    print("    ✓ Z_diesel: Diesel-Kosten")
    
    # Z_toll: Mautkosten (nur Diesel)
    z_toll = OptimizationConfig.NUM_DAYS * OptimizationConfig.TOLL_PRICE * quicksum(
        y[v, r] * params['distance_toll'][r]
        for v in sets['V_D'] for r in sets['R']
    )
    print("    ✓ Z_toll: Maut-Kosten")
    
    # Z_revenue: THG-Erlöse
    z_revenue = quicksum(
        x[v] * params['e_thg_revenue'][sets['model_map_E'][v]]
        for v in sets['V_E']
    )
    print("    ✓ Z_revenue: THG-Erlöse")
    
    # Gesamtkosten
    total_cost = (z_vehicles_e + z_vehicles_d + z_infrastructure + z_grid + 
                  z_storage + z_electricity + z_diesel + z_toll - z_revenue)
    
    model.setObjective(total_cost, "minimize")
    print("    ✓ Zielfunktion: min Z (Total Cost of Ownership)")
    
    # ========================================================================
    # NEBENBEDINGUNGEN
    # ========================================================================
    print("\n[3/6] Nebenbedingungen werden hinzugefügt...")
    
    constraint_count = 0
    
    # (1) Tourenzuordnung: Jede Tour genau einmal
    for r in sets['R']:
        model.addCons(quicksum(y[v, r] for v in sets['V']) == 1, 
                      name=f"tour_assignment_{r}")
    constraint_count += len(sets['R'])
    print(f"    ✓ (1) {len(sets['R'])} Tourenzuordnungs-Constraints")
    
    # (2) Fahrzeugexistenz
    for v in sets['V']:
        for r in sets['R']:
            model.addCons(y[v, r] <= x[v], name=f"vehicle_exists_{v}_{r}")
    constraint_count += len(sets['V']) * len(sets['R'])
    print(f"    ✓ (2) {len(sets['V']) * len(sets['R'])} Fahrzeugexistenz-Constraints")
    
    # (3) Keine zeitliche Überlappung
    for v in sets['V']:
        for t in sets['T']:
            # Finde alle Routen, die zum Zeitpunkt t aktiv sind
            active_routes = [r for r in sets['R'] 
                           if params['tour_start'][r] <= t <= params['tour_end'][r]]
            if active_routes:
                model.addCons(quicksum(y[v, r] for r in active_routes) <= 1,
                             name=f"no_overlap_{v}_{t}")
                constraint_count += 1
    print(f"    ✓ (3) Zeitliche Überlappungs-Constraints hinzugefügt")
    
    # (4) Ladeinfrastruktur - Maximal 3 Ladesäulen INSGESAMT
    model.addCons(quicksum(z[c] for c in sets['C']) <= 3,
                 name="max_total_chargers")
    constraint_count += 1
    print(f"    ✓ (4) 1 Gesamt-Ladeinfrastruktur-Constraint (max. 3 Säulen insgesamt)")
    
    
    # (5) NEU: 2 Ladepunkte pro Säule - Pro Typ dürfen max 2*z[c] Trucks gleichzeitig hängen
    for t in sets['T']:
        for c in sets['C']:
            model.addCons(
                quicksum(a[v, t, c] for v in sets['V_E']) <= 2 * z[c],
                name=f"spot_capacity_{c}_{t}"
            )
    constraint_count += len(sets['T']) * len(sets['C'])
    print(f"    ✓ (5) {len(sets['T']) * len(sets['C'])} Ladepunkt-Kapazitäts-Constraints (2 pro Säule)")


    # (5a) NEU: Exakt ein Typ wenn geladen - Truck muss sich für einen Säulentyp entscheiden
    for v in sets['V_E']:
        for t in sets['T']:
            model.addCons(
                quicksum(a[v, t, c] for c in sets['C']) == w[v, t],
                name=f"one_type_if_charging_{v}_{t}"
            )
    constraint_count += len(sets['V_E']) * len(sets['T'])
    print(f"    ✓ (5a) {len(sets['V_E']) * len(sets['T'])} Typ-Zuordnungs-Constraints (exakt ein Typ)")
    
    # (5b) NEU: Leistung eines Trucks ist Summe seiner Typ-Leistungen
    for v in sets['V_E']:
        for t in sets['T']:
            model.addCons(
                quicksum(p_by[v, t, c] for c in sets['C']) == p_charge[v, t],
                name=f"power_sum_{v}_{t}"
            )
    constraint_count += len(sets['V_E']) * len(sets['T'])
    print(f"    ✓ (5b) {len(sets['V_E']) * len(sets['T'])} Leistungssummen-Constraints")
    
    # (6) Keine gleichzeitige Fahrt und Ladung (VEREINFACHT)
    for v in sets['V_E']:
        for t in sets['T']:
            active_routes = [r for r in sets['R'] 
                           if params['tour_start'][r] <= t <= params['tour_end'][r]]
            model.addCons(
                quicksum(y[v, r] for r in active_routes) + w[v, t] <= 1,
                name=f"no_drive_and_charge_{v}_{t}"
            )
    constraint_count += len(sets['V_E']) * len(sets['T'])
    print(f"    ✓ (6) {len(sets['V_E']) * len(sets['T'])} Fahrt-Ladung-Exklusivitäts-Constraints")
    
    # (7) NEU: Typ-Zuordnung schaltet Leistung frei - pro Truck max P_c über diesen Typ
    for v in sets['V_E']:
        model_type = sets['model_map_E'][v]
        max_vehicle_power = params['e_charge_power'][model_type]
        for t in sets['T']:
            for c in sets['C']:
                # Ein Truck darf über Typ c höchstens P_c ziehen (wenn er dran hängt)
                # Zusätzlich begrenzt durch Fahrzeug-Maximum
                max_power = min(params['charger_power'][c], max_vehicle_power)
                model.addCons(
                    p_by[v, t, c] <= max_power * a[v, t, c],
                    name=f"type_power_limit_{v}_{t}_{c}"
                )
    constraint_count += len(sets['V_E']) * len(sets['T']) * len(sets['C'])
    print(f"    ✓ (7) {len(sets['V_E']) * len(sets['T']) * len(sets['C'])} Typ-Leistungs-Constraints")
    
    # (9) NEU: Gesamtleistung pro Typ - alle Trucks teilen sich die verfügbare Leistung
    for t in sets['T']:
        for c in sets['C']:
            model.addCons(
                quicksum(p_by[v, t, c] for v in sets['V_E']) 
                <= params['charger_power'][c] * z[c],
                name=f"charger_power_limit_{c}_{t}"
            )
    constraint_count += len(sets['T']) * len(sets['C'])
    print(f"    ✓ (9) {len(sets['T']) * len(sets['C'])} Ladesäulen-Leistungs-Constraints (Leistungsteilung)")
    
    # (10) Operative Depot-Regeln: Nacht-Lade-Kontinuität (18:00-6:00) - VEREINFACHT
    # Zeitschritte: 18:00 = t=72, 6:00 = t=24 (nächster Tag)
    # Wenn ein E-Lkw im Depot ist (nicht fährt) und lädt, muss er kontinuierlich laden
    night_start = 88  # 22:00 Uhr
    night_end = 24    # 6:00 Uhr (nächster Tag)
    
    for v in sets['V_E']:
        # Nachtzeitraum: 22:00 bis 23:45 (t=88 bis t=95)
        for t in range(night_start, OptimizationConfig.NUM_TIMESTEPS - 1):
            active_routes_t1 = [r for r in sets['R'] 
                               if params['tour_start'][r] <= t+1 <= params['tour_end'][r]]
            # Wenn in t lädt und in t+1 im Depot → muss auch laden
            model.addCons(
                w[v, t] <= w[v, t+1] + quicksum(y[v, r] for r in active_routes_t1),
                name=f"night_continuity_{v}_{t}"
            )
            constraint_count += 1
        
        # Nachtzeitraum: 0:00 bis 5:45 (t=0 bis t=23)
        for t in range(0, night_end - 1):
            active_routes_t1 = [r for r in sets['R'] 
                               if params['tour_start'][r] <= t+1 <= params['tour_end'][r]]
            model.addCons(
                w[v, t] <= w[v, t+1] + quicksum(y[v, r] for r in active_routes_t1),
                name=f"night_continuity_early_{v}_{t}"
            )
            constraint_count += 1
    
    print(f"    ✓ (10) {len(sets['V_E']) * (OptimizationConfig.NUM_TIMESTEPS - night_start - 1 + night_end - 1)} Nacht-Lade-Kontinuitäts-Constraints (VEREINFACHT)")
    
    # (11) Batteriebilanz - SOC-Dynamik
    for v in sets['V_E']:
        model_type = sets['model_map_E'][v]
        energy_cons = params['e_energy_cons'][model_type]
        
        for t in sets['T'][:-1]:  # Alle außer dem letzten
            # Energieverbrauch während aktiver Touren
            # Vereinfachung: Gleichmäßiger Verbrauch über Tourdauer
            energy_used = 0
            for r in sets['R']:
                if params['tour_start'][r] <= t <= params['tour_end'][r]:
                    tour_duration = (params['tour_end'][r] - params['tour_start'][r] + 1)
                    energy_per_step = (params['distance_total'][r] * energy_cons) / tour_duration
                    energy_used += y[v, r] * energy_per_step
            
            # SOC-Bilanz (VEREINFACHT - keine Spot-Summierung mehr)
            model.addCons(
                soc[v, t+1] == soc[v, t] + 
                OptimizationConfig.DELTA_T * (p_charge[v, t] - energy_used / OptimizationConfig.DELTA_T),
                name=f"battery_balance_{v}_{t}"
            )
    constraint_count += len(sets['V_E']) * (len(sets['T']) - 1)
    print(f"    ✓ (11) {len(sets['V_E']) * (len(sets['T']) - 1)} Batteriebilanz-Constraints")
    
    # (12) Zyklischer SOC
    for v in sets['V_E']:
        model.addCons(soc[v, 0] == soc[v, len(sets['T'])-1], 
                     name=f"cyclic_soc_{v}")
    constraint_count += len(sets['V_E'])
    print(f"    ✓ (12) {len(sets['V_E'])} Zyklische-SOC-Constraints")
    
    # (13) SOC-Grenzen
    for v in sets['V_E']:
        model_type = sets['model_map_E'][v]
        battery_cap = params['e_battery_cap'][model_type]
        for t in sets['T']:
            model.addCons(soc[v, t] <= battery_cap * x[v],
                         name=f"soc_upper_{v}_{t}")
    constraint_count += len(sets['V_E']) * len(sets['T'])
    print(f"    ✓ (13) {len(sets['V_E']) * len(sets['T'])} SOC-Grenzen-Constraints")
    
    # (14) Leistungsbilanz: Grid + Speicher-Entladung = Laden + Speicher-Ladung (VEREINFACHT)
    for t in sets['T']:
        total_charging = quicksum(p_charge[v, t] for v in sets['V_E'])
        model.addCons(
            p_grid[t] + p_stor_discharge[t] == total_charging + p_stor_charge[t],
            name=f"power_balance_{t}"
        )
    constraint_count += len(sets['T'])
    print(f"    ✓ (14) {len(sets['T'])} Leistungsbilanz-Constraints")
    
    # (15) Netzanschlussgrenze
    for t in sets['T']:
        model.addCons(
            p_grid[t] <= OptimizationConfig.GRID_BASE_POWER + 
                        u_grid * OptimizationConfig.GRID_UPGRADE_POWER,
            name=f"grid_limit_{t}"
        )
    constraint_count += len(sets['T'])
    print(f"    ✓ (15) {len(sets['T'])} Netzanschluss-Constraints")
    
    # (16) Peak-Definition
    for t in sets['T']:
        model.addCons(p_peak >= p_grid[t], name=f"peak_def_{t}")
    constraint_count += len(sets['T'])
    print(f"    ✓ (16) {len(sets['T'])} Peak-Definitions-Constraints")
    
    # (17) Speicher-Leistungsgrenze
    for t in sets['T']:
        model.addCons(p_stor_charge[t] <= P_stor, name=f"stor_charge_lim_{t}")
        model.addCons(p_stor_discharge[t] <= P_stor, name=f"stor_discharge_lim_{t}")
    constraint_count += 2 * len(sets['T'])
    print(f"    ✓ (17) {2 * len(sets['T'])} Speicher-Leistungs-Constraints")
    
    # (18) Speicher-Bilanz
    eta_c = OptimizationConfig.STORAGE_ETA_CHARGE
    eta_d = OptimizationConfig.STORAGE_ETA_DISCHARGE
    for t in sets['T'][:-1]:
        model.addCons(
            soc_stor[t+1] == soc_stor[t] + 
            OptimizationConfig.DELTA_T * (eta_c * p_stor_charge[t] - p_stor_discharge[t] / eta_d),
            name=f"storage_balance_{t}"
        )
    constraint_count += len(sets['T']) - 1
    print(f"    ✓ (18) {len(sets['T']) - 1} Speicher-Bilanz-Constraints")
    
    # (19) Zyklischer Speicher-SOC
    model.addCons(soc_stor[0] == soc_stor[len(sets['T'])-1], name="cyclic_storage_soc")
    constraint_count += 1
    print(f"    ✓ (19) 1 Zyklischer-Speicher-SOC-Constraint")
    
    # (20) Speicher-SOC-Grenzen mit DoD
    for t in sets['T']:
        model.addCons(soc_stor[t] >= OptimizationConfig.STORAGE_DOD_MIN * E_stor,
                     name=f"storage_soc_lower_{t}")
        model.addCons(soc_stor[t] <= E_stor, name=f"storage_soc_upper_{t}")
    constraint_count += 2 * len(sets['T'])
    print(f"    ✓ (20) {2 * len(sets['T'])} Speicher-SOC-Grenzen-Constraints")
    
    print(f"\n    GESAMT: {constraint_count} Constraints hinzugefügt")
    
    # ========================================================================
    # MODELL-STATISTIK
    # ========================================================================
    print("\n[4/6] Modell-Statistik:")
    print(f"    • Variablen: {model.getNVars()}")
    print(f"    • Binärvariablen: {len(x) + len(y) + len(w) + 1}")  # +1 für u_grid
    print(f"    • Ganzzahlige Variablen: {len(z)}")
    print(f"    • Kontinuierliche Variablen: {model.getNVars() - len(x) - len(y) - len(w) - len(z) - 1}")
    print(f"    • Constraints: {constraint_count}")
    
    return model, {
        'x': x, 'y': y, 'z': z, 'w': w, 
        'p_charge': p_charge, 'a': a, 'p_by': p_by, 'soc': soc,
        'p_grid': p_grid, 'p_peak': p_peak, 'u_grid': u_grid,
        'P_stor': P_stor, 'E_stor': E_stor,
        'p_stor_charge': p_stor_charge, 'p_stor_discharge': p_stor_discharge,
        'soc_stor': soc_stor
    }


# ============================================================================
# OPTIMIERUNG UND ERGEBNIS-AUSWERTUNG
# ============================================================================

def optimize_and_analyze(model, variables, sets, params):
    """
    Optimiert das Modell und wertet die Ergebnisse aus
    """
    print("\n[5/6] Optimierung wird gestartet...")
    print("="*80)
    
    start_time = datetime.now()
    model.optimize()
    end_time = datetime.now()
    
    solve_time = (end_time - start_time).total_seconds()
    
    print("="*80)
    print("\n[6/6] ERGEBNIS-AUSWERTUNG")
    print("="*80)
    
    status = model.getStatus()
    print(f"\nOptimierungsstatus: {status}")
    print(f"Lösungszeit: {solve_time:.2f} Sekunden")
    
    # Erweiterte Statusanzeige für alle Fälle
    print(f"\n{'═'*80}")
    print("OPTIMIERUNGSERGEBNIS")
    print(f"{'═'*80}")
    
    # Prüfe ob überhaupt eine Lösung gefunden wurde
    n_sols = model.getNSols()
    print(f"Anzahl gefundener Lösungen: {n_sols}")
    
    if n_sols > 0:
        # Beste gefundene Lösung auslesen
        obj_value = model.getObjVal()
        primal_bound = model.getPrimalbound()
        dual_bound = model.getDualbound()
        gap = model.getGap()
        
        print(f"\n{'─'*80}")
        print("BESTE GEFUNDENE LÖSUNG:")
        print(f"{'─'*80}")
        print(f"Zielfunktionswert (Primal Bound): {primal_bound:,.2f} €/Jahr")
        print(f"Untere Schranke (Dual Bound):     {dual_bound:,.2f} €/Jahr")
        print(f"Optimierungslücke (Gap):          {gap*100:.4f}%")
        
        # Statusabhängige Meldung
        if status == "optimal":
            print(f"\n✓ OPTIMAL: Beste Lösung ist nachweislich optimal!")
        elif status == "timelimit":
            print(f"\n⏱ ZEITLIMIT: Optimierung wurde durch Zeitlimit gestoppt.")
            print(f"   Die oben gezeigte Lösung ist die beste gefundene Lösung,")
            print(f"   aber möglicherweise nicht optimal (Gap: {gap*100:.4f}%).")
        elif status == "gaplimit":
            print(f"\n✓ GAP-LIMIT: Gewünschte Optimalitätslücke erreicht!")
            print(f"   Die Lösung ist höchstens {gap*100:.4f}% vom Optimum entfernt.")
        elif status == "bestsollimit":
            print(f"\n⏹ BEST-SOLUTION-LIMIT: Maximale Anzahl Lösungen erreicht.")
        else:
            print(f"\n⚠ STATUS: {status}")
            print(f"   Eine gültige Lösung wurde gefunden.")
        
        print(f"{'─'*80}")
        
        # Flottenzusammensetzung
        print("\n" + "="*80)
        print("FLOTTENZUSAMMENSETZUNG")
        print("="*80)
        
        # E-Lkw
        e_fleet = {}
        for v in sets['V_E']:
            if model.getVal(variables['x'][v]) > 0.5:
                model_type = sets['model_map_E'][v]
                e_fleet[model_type] = e_fleet.get(model_type, 0) + 1
        
        print("\nElektro-Lkw:")
        if e_fleet:
            for model_type, count in sorted(e_fleet.items()):
                print(f"  • {model_type}: {count} Fahrzeuge")
        else:
            print("  • Keine E-Lkw beschafft")
        
        # Diesel-Lkw
        d_fleet = {}
        for v in sets['V_D']:
            if model.getVal(variables['x'][v]) > 0.5:
                model_type = sets['model_map_D'][v]
                d_fleet[model_type] = d_fleet.get(model_type, 0) + 1
        
        print("\nDiesel-Lkw:")
        if d_fleet:
            for model_type, count in sorted(d_fleet.items()):
                print(f"  • {model_type}: {count} Fahrzeuge")
        else:
            print("  • Keine Diesel-Lkw beschafft")
        
        # Ladeinfrastruktur
        print("\n" + "="*80)
        print("LADEINFRASTRUKTUR")
        print("="*80)
        
        for c in sets['C']:
            num_chargers = round(model.getVal(variables['z'][c]))
            if num_chargers > 0:
                print(f"  • {c}: {num_chargers} Ladesäulen")
        
        # Netzanschluss
        print("\n" + "="*80)
        print("NETZANSCHLUSS")
        print("="*80)
        
        grid_upgrade = model.getVal(variables['u_grid']) > 0.5
        peak_power = model.getVal(variables['p_peak'])
        
        if grid_upgrade:
            total_capacity = OptimizationConfig.GRID_BASE_POWER + OptimizationConfig.GRID_UPGRADE_POWER
            print(f"  • Netzerweiterung: JA (Gesamt: {total_capacity:.0f} kW)")
        else:
            print(f"  • Netzerweiterung: NEIN (Basis: {OptimizationConfig.GRID_BASE_POWER:.0f} kW)")
        
        print(f"  • Jahreshöchstlast: {peak_power:.2f} kW")
        
        # Batteriespeicher
        print("\n" + "="*80)
        print("BATTERIESPEICHER")
        print("="*80)
        
        stor_power = model.getVal(variables['P_stor'])
        stor_energy = model.getVal(variables['E_stor'])
        
        if stor_power > 0.1 or stor_energy > 0.1:
            print(f"  • Leistung: {stor_power:.2f} kW")
            print(f"  • Kapazität: {stor_energy:.2f} kWh")
            if stor_energy > 0:
                print(f"  • C-Rate: {stor_power/stor_energy:.2f}")
        else:
            print("  • Kein Speicher installiert")
        
        # Kostenaufschlüsselung
        print("\n" + "="*80)
        print("KOSTENAUFSCHLÜSSELUNG")
        print("="*80)
        
        # E-Lkw Kosten
        e_cost = sum(
            model.getVal(variables['x'][v]) * 
            (params['e_capex'][sets['model_map_E'][v]] + params['e_opex'][sets['model_map_E'][v]])
            for v in sets['V_E']
        )
        print(f"\nE-Lkw (CAPEX + OPEX):        {e_cost:>15,.2f} €/a")
        
        # Diesel Kosten
        d_cost = sum(
            model.getVal(variables['x'][v]) * 
            (params['d_capex'][sets['model_map_D'][v]] + 
             params['d_opex'][sets['model_map_D'][v]] +
             params['d_tax'][sets['model_map_D'][v]])
            for v in sets['V_D']
        )
        print(f"Diesel-Lkw (CAPEX + OPEX):   {d_cost:>15,.2f} €/a")
        
        # Ladeinfrastruktur
        infra_cost = sum(
            model.getVal(variables['z'][c]) * 
            (params['charger_capex'][c] + params['charger_opex'][c])
            for c in sets['C']
        )
        print(f"Ladeinfrastruktur:           {infra_cost:>15,.2f} €/a")
        
        # Netzanschluss
        grid_cost = (OptimizationConfig.GRID_BASE_FEE + 
                    model.getVal(variables['u_grid']) * OptimizationConfig.GRID_UPGRADE_COST)
        print(f"Netzanschluss:               {grid_cost:>15,.2f} €/a")
        
        # Speicher
        storage_cost = (stor_power * (OptimizationConfig.STORAGE_CAPEX_POWER + 
                                      OptimizationConfig.STORAGE_OPEX_POWER) +
                       stor_energy * (OptimizationConfig.STORAGE_CAPEX_ENERGY + 
                                      OptimizationConfig.STORAGE_OPEX_ENERGY))
        print(f"Batteriespeicher:            {storage_cost:>15,.2f} €/a")
        
        # Stromkosten
        total_energy = sum(
            model.getVal(variables['p_grid'][t]) * OptimizationConfig.DELTA_T 
            for t in sets['T']
        ) * OptimizationConfig.NUM_DAYS
        
        electricity_work_cost = total_energy * OptimizationConfig.ELECTRICITY_PRICE
        electricity_peak_cost = peak_power * OptimizationConfig.PEAK_PRICE 
        electricity_cost = electricity_work_cost + electricity_peak_cost
        
        print(f"Strom (Arbeitspreis):        {electricity_work_cost:>15,.2f} €/a")
        print(f"Strom (Leistungspreis):      {electricity_peak_cost:>15,.2f} €/a")
        
        # Diesel
        diesel_cost = OptimizationConfig.NUM_DAYS * OptimizationConfig.DIESEL_PRICE * sum(
            model.getVal(variables['y'][v, r]) * 
            params['d_fuel_cons'][sets['model_map_D'][v]] * 
            params['distance_total'][r]
            for v in sets['V_D'] for r in sets['R']
        )
        print(f"Diesel:                      {diesel_cost:>15,.2f} €/a")
        
        # Maut
        toll_cost = OptimizationConfig.NUM_DAYS * OptimizationConfig.TOLL_PRICE * sum(
            model.getVal(variables['y'][v, r]) * params['distance_toll'][r]
            for v in sets['V_D'] for r in sets['R']
        )
        print(f"Maut:                        {toll_cost:>15,.2f} €/a")
        
        # THG-Erlöse
        thg_revenue = sum(
            model.getVal(variables['x'][v]) * params['e_thg_revenue'][sets['model_map_E'][v]]
            for v in sets['V_E']
        )
        print(f"THG-Erlöse:                 {-thg_revenue:>15,.2f} €/a")
        
        print(f"\n{'─'*80}")
        print(f"GESAMTSUMME:                 {obj_value:>15,.2f} €/a")
        print(f"{'─'*80}")
        
        # ====================================================================
        # ERWEITERTETS REPORTING - TOURENZUORDNUNG
        # ====================================================================
        print("\n" + "="*80)
        print("DETAILLIERTE TOURENZUORDNUNG")
        print("="*80)
        
        # Sammle Tourenzuordnungen
        tour_assignments = {}
        for r in sets['R']:
            for v in sets['V']:
                if model.getVal(variables['y'][v, r]) > 0.5:
                    is_electric = v in sets['V_E']
                    model_type = sets['model_map_E'][v] if is_electric else sets['model_map_D'][v]
                    tour_assignments[r] = {
                        'vehicle': v,
                        'type': model_type,
                        'is_electric': is_electric,
                        'distance': params['distance_total'][r],
                        'distance_toll': params['distance_toll'][r],
                        'start_time': params['tour_start'][r],
                        'end_time': params['tour_end'][r]
                    }
                    break
        
        # Ausgabe sortiert nach Route
        print("\nRoute | Fahrzeug-Typ | Fahrzeug-ID | Typ | Distanz | Maut-km | Start | Ende")
        print("-"*80)
        for r in sorted(sets['R']):
            if r in tour_assignments:
                ta = tour_assignments[r]
                vehicle_type = "E-Lkw" if ta['is_electric'] else "Diesel"
                start_h = ta['start_time'] * 0.25
                end_h = ta['end_time'] * 0.25
                print(f"{r:5s} | {ta['type']:12s} | {ta['vehicle']:15s} | {vehicle_type:6s} | "
                      f"{ta['distance']:7.1f} | {ta['distance_toll']:7.1f} | "
                      f"{start_h:5.2f} | {end_h:5.2f}")
        
        # Statistik
        e_tours = sum(1 for ta in tour_assignments.values() if ta['is_electric'])
        d_tours = len(tour_assignments) - e_tours
        print(f"\nZusammenfassung: {e_tours} Touren elektrisch, {d_tours} Touren Diesel")
        
        # ====================================================================
        # ERWEITERTETS REPORTING - FAHRZEUGAUSLASTUNG
        # ====================================================================
        print("\n" + "="*80)
        print("FAHRZEUGAUSLASTUNG")
        print("="*80)
        
        vehicle_usage = {}
        for v in sets['V']:
            if model.getVal(variables['x'][v]) > 0.5:
                is_electric = v in sets['V_E']
                model_type = sets['model_map_E'][v] if is_electric else sets['model_map_D'][v]
                tours = [r for r in sets['R'] if model.getVal(variables['y'][v, r]) > 0.5]
                total_dist = sum(params['distance_total'][r] for r in tours)
                vehicle_usage[v] = {
                    'type': model_type,
                    'is_electric': is_electric,
                    'num_tours': len(tours),
                    'total_distance': total_dist,
                    'tours': tours
                }
        
        print("\nFahrzeug-ID | Typ | Modell | Anzahl Touren | Gesamt-km | Touren")
        print("-"*80)
        for v in sorted(vehicle_usage.keys()):
            vu = vehicle_usage[v]
            vehicle_type = "E-Lkw" if vu['is_electric'] else "Diesel"
            tours_str = ", ".join(vu['tours'][:5])
            if len(vu['tours']) > 5:
                tours_str += f", ... (+{len(vu['tours'])-5})"
            print(f"{v:15s} | {vehicle_type:6s} | {vu['type']:12s} | "
                  f"{vu['num_tours']:13d} | {vu['total_distance']:9.1f} | {tours_str}")
        
        # ====================================================================
        # ERWEITERTETS REPORTING - E-LKW LADEVERHALTEN
        # ====================================================================
        if e_fleet:
            print("\n" + "="*80)
            print("E-LKW LADEVERHALTEN (Übersicht)")
            print("="*80)
            
            for v in sets['V_E']:
                if model.getVal(variables['x'][v]) > 0.5:
                    model_type = sets['model_map_E'][v]
                    battery_cap = params['e_battery_cap'][model_type]
                    
                    # Sammle Ladezeiten
                    charging_periods = []
                    in_charging = False
                    charge_start = None
                    
                    for t in sets['T']:
                        is_charging = model.getVal(variables['w'][v, t]) > 0.5
                        if is_charging and not in_charging:
                            charge_start = t
                            in_charging = True
                        elif not is_charging and in_charging:
                            charging_periods.append((charge_start, t-1))
                            in_charging = False
                    
                    if in_charging:
                        charging_periods.append((charge_start, sets['T'][-1]))
                    
                    # SOC Min/Max
                    soc_values = [model.getVal(variables['soc'][v, t]) for t in sets['T']]
                    soc_min = min(soc_values)
                    soc_max = max(soc_values)
                    
                    # Gesamtenergie geladen
                    total_energy_charged = sum(
                        model.getVal(variables['p_charge'][v, t]) * OptimizationConfig.DELTA_T
                        for t in sets['T']
                    )
                    
                    print(f"\n{v} ({model_type}):")
                    print(f"  • Batteriekapazität: {battery_cap:.1f} kWh")
                    print(f"  • SOC-Bereich: {soc_min:.1f} - {soc_max:.1f} kWh "
                          f"({soc_min/battery_cap*100:.1f}% - {soc_max/battery_cap*100:.1f}%)")
                    print(f"  • Energie geladen (pro Tag): {total_energy_charged:.1f} kWh")
                    print(f"  • Anzahl Ladeperioden: {len(charging_periods)}")
                    
                    if charging_periods:
                        print(f"  • Ladezeiten:")
                        for start, end in charging_periods[:5]:
                            start_h = start * 0.25
                            end_h = (end + 1) * 0.25
                            duration = (end - start + 1) * 0.25
                            avg_power = sum(model.getVal(variables['p_charge'][v, t]) 
                                          for t in range(start, end+1)) / (end - start + 1)
                            print(f"      {start_h:05.2f}h - {end_h:05.2f}h ({duration:.2f}h) @ Ø {avg_power:.1f} kW")
                        if len(charging_periods) > 5:
                            print(f"      ... (+{len(charging_periods)-5} weitere)")
        
        # ====================================================================
        # ERWEITERTETS REPORTING - NETZBEZUG PROFIL
        # ====================================================================
        print("\n" + "="*80)
        print("NETZBEZUGSPROFIL")
        print("="*80)
        
        # Stundenweise Zusammenfassung (Mittelwert über 4 Zeitschritte)
        print("\nStunde | Ø Netzbezug | Max Netzbezug | Ø Laden | Ø Speicher (L/E)")
        print("-"*80)
        
        for hour in range(24):
            timesteps = range(hour*4, (hour+1)*4)
            avg_grid = sum(model.getVal(variables['p_grid'][t]) for t in timesteps) / 4
            max_grid = max(model.getVal(variables['p_grid'][t]) for t in timesteps)
            avg_charging = sum(
                sum(model.getVal(variables['p_charge'][v, t]) for v in sets['V_E'])
                for t in timesteps
            ) / 4
            avg_stor_charge = sum(model.getVal(variables['p_stor_charge'][t]) for t in timesteps) / 4
            avg_stor_discharge = sum(model.getVal(variables['p_stor_discharge'][t]) for t in timesteps) / 4
            
            print(f"{hour:2d}:00  | {avg_grid:11.2f} | {max_grid:13.2f} | "
                  f"{avg_charging:7.2f} | {avg_stor_charge:6.2f}/{avg_stor_discharge:.2f}")
        
        # ====================================================================
        # ERWEITERTETS REPORTING - SPEICHERNUTZUNG
        # ====================================================================
        if stor_power > 0.1 or stor_energy > 0.1:
            print("\n" + "="*80)
            print("SPEICHERNUTZUNG")
            print("="*80)
            
            # Zyklen zählen (vereinfacht)
            total_energy_charged = sum(
                model.getVal(variables['p_stor_charge'][t]) * OptimizationConfig.DELTA_T
                for t in sets['T']
            )
            total_energy_discharged = sum(
                model.getVal(variables['p_stor_discharge'][t]) * OptimizationConfig.DELTA_T
                for t in sets['T']
            )
            
            soc_stor_values = [model.getVal(variables['soc_stor'][t]) for t in sets['T']]
            soc_stor_min = min(soc_stor_values)
            soc_stor_max = max(soc_stor_values)
            
            print(f"\n  • Tägliche Ladung: {total_energy_charged:.2f} kWh")
            print(f"  • Tägliche Entladung: {total_energy_discharged:.2f} kWh")
            print(f"  • Wirkungsgrad (Tagesschnitt): {(total_energy_discharged/total_energy_charged*100) if total_energy_charged > 0 else 0:.1f}%")
            print(f"  • SOC-Bereich: {soc_stor_min:.1f} - {soc_stor_max:.1f} kWh "
                  f"({soc_stor_min/stor_energy*100:.1f}% - {soc_stor_max/stor_energy*100:.1f}%)")
            
            # Vollzyklen pro Tag (als Schätzung)
            full_cycles = total_energy_discharged / stor_energy if stor_energy > 0 else 0
            print(f"  • Vollzyklen pro Tag: {full_cycles:.2f}")
            print(f"  • Vollzyklen pro Jahr: {full_cycles * OptimizationConfig.NUM_DAYS:.0f}")
        
        # ====================================================================
        # ERWEITERTETS REPORTING - LADESÄULENNUTZUNG
        # ====================================================================
        print("\n" + "="*80)
        print("LADESÄULENNUTZUNG")
        print("="*80)
        
        for c in sets['C']:
            num_chargers = round(model.getVal(variables['z'][c]))
            if num_chargers > 0:
                # Auslastung berechnen
                total_power_used = sum(
                    sum(model.getVal(variables['p_by'][v, t, c]) for v in sets['V_E'])
                    for t in sets['T']
                )
                max_power_available = params['charger_power'][c] * num_chargers * len(sets['T'])
                utilization = (total_power_used / max_power_available * 100) if max_power_available > 0 else 0
                
                # Maximale gleichzeitige Nutzung
                max_concurrent = max(
                    sum(1 for v in sets['V_E'] if model.getVal(variables['a'][v, t, c]) > 0.5)
                    for t in sets['T']
                )
                
                # Durchschnittliche Leistung wenn genutzt
                periods_used = sum(
                    1 for t in sets['T']
                    if sum(model.getVal(variables['p_by'][v, t, c]) for v in sets['V_E']) > 0.1
                )
                avg_power_when_used = (total_power_used / periods_used) if periods_used > 0 else 0
                
                print(f"\n{c}:")
                print(f"  • Anzahl installiert: {num_chargers}")
                print(f"  • Max. Leistung: {params['charger_power'][c]:.0f} kW pro Säule")
                print(f"  • Ladepunkte: {params['charger_spots'][c]} pro Säule")
                print(f"  • Auslastung (Leistung): {utilization:.1f}%")
                print(f"  • Max. gleichzeitige Nutzung: {max_concurrent} Fahrzeuge")
                print(f"  • Genutzte Zeitschritte: {periods_used}/{len(sets['T'])} ({periods_used/len(sets['T'])*100:.1f}%)")
                print(f"  • Ø Leistung wenn genutzt: {avg_power_when_used:.1f} kW")
        
        # ====================================================================
        # ERWEITERTETS REPORTING - UMWELTKENNZAHLEN
        # ====================================================================
        print("\n" + "="*80)
        print("UMWELT- UND ENERGIEKENNZAHLEN")
        print("="*80)
        
        # Elektrische Energie
        total_electric_energy = sum(
            sum(model.getVal(variables['p_charge'][v, t]) * OptimizationConfig.DELTA_T 
                for t in sets['T'])
            for v in sets['V_E']
        ) * OptimizationConfig.NUM_DAYS
        
        # Diesel-Energie
        total_diesel_energy = OptimizationConfig.NUM_DAYS * sum(
            model.getVal(variables['y'][v, r]) * 
            params['d_fuel_cons'][sets['model_map_D'][v]] * 
            params['distance_total'][r]
            for v in sets['V_D'] for r in sets['R']
        ) * 10  # kWh pro Liter Diesel (Heizwert ca. 10 kWh/L)
        
        # Distanzen
        total_electric_km = sum(
            model.getVal(variables['y'][v, r]) * params['distance_total'][r]
            for v in sets['V_E'] for r in sets['R']
        ) * OptimizationConfig.NUM_DAYS
        
        total_diesel_km = sum(
            model.getVal(variables['y'][v, r]) * params['distance_total'][r]
            for v in sets['V_D'] for r in sets['R']
        ) * OptimizationConfig.NUM_DAYS
        
        total_km = total_electric_km + total_diesel_km
        
        print(f"\nJährliche Fahrleistung:")
        print(f"  • Elektrisch: {total_electric_km:,.0f} km ({total_electric_km/total_km*100:.1f}%)")
        print(f"  • Diesel: {total_diesel_km:,.0f} km ({total_diesel_km/total_km*100:.1f}%)")
        print(f"  • Gesamt: {total_km:,.0f} km")
        
        print(f"\nJährlicher Energieverbrauch:")
        print(f"  • Strom (Fahrzeuge): {total_electric_energy:,.0f} kWh")
        print(f"  • Diesel (Äquivalent): {total_diesel_energy:,.0f} kWh")
        print(f"  • Gesamt: {total_electric_energy + total_diesel_energy:,.0f} kWh")
        
        # Spezifischer Verbrauch
        if total_electric_km > 0:
            print(f"\nSpezifischer Energieverbrauch:")
            print(f"  • E-Lkw: {total_electric_energy/total_electric_km:.2f} kWh/km")
        if total_diesel_km > 0:
            diesel_liters = diesel_cost / (OptimizationConfig.DIESEL_PRICE * OptimizationConfig.NUM_DAYS)
            print(f"  • Diesel: {diesel_liters*100/total_diesel_km:.2f} L/100km")
        
        # ====================================================================
        # ZUSÄTZLICHE OPTIMIERUNGSINFORMATIONEN
        # ====================================================================
        print(f"\n{'═'*80}")
        print("OPTIMIERUNGSDETAILS")
        print(f"{'═'*80}")
        print(f"Status:                      {status}")
        print(f"Lösungszeit:                 {solve_time:.2f} Sekunden")
        print(f"Gap:                         {gap*100:.4f}%")
        print(f"Primal Bound:                {primal_bound:,.2f} €/a")
        print(f"Dual Bound:                  {dual_bound:,.2f} €/a")
        print(f"Anzahl Lösungen:             {n_sols}")
        
        # Speichere Lösung
        model.writeBestSol("best_solution.sol")
        print(f"\n✓ Beste Lösung gespeichert in: best_solution.sol")
        
    else:
        print("\n⚠ Keine Lösung gefunden!")
        print(f"Status: {status}")
        print(f"\nMögliche Ursachen:")
        print(f"  • Modell ist infeasible (keine gültige Lösung existiert)")
        print(f"  • Modell ist unbounded")
        print(f"  • Optimierung wurde zu früh abgebrochen")
    
    return model


# ============================================================================
# HAUPTPROGRAMM
# ============================================================================

def main():
    """Hauptfunktion - orchestriert den gesamten Optimierungsprozess"""
    
    print("\n" + "="*80)
    print(" "*15 + "DEPOT-ELEKTRIFIZIERUNG - MILP-OPTIMIERUNG")
    print("="*80)
    print("\nMathematisches Modell zur Minimierung der Total Cost of Ownership (TCO)")
    print("fuer die Elektrifizierung eines Logistikdepots mit gemischter Flotte.\n")
    
    # Schritt 1: Daten laden
    print("\n" + "-"*80)
    print("SCHRITT 1: DATENIMPORT")
    print("-"*80)
    data = load_data()
    
    # Schritt 2: Indexmengen aufbauen
    print("\n" + "-"*80)
    print("SCHRITT 2: INDEXMENGEN-AUFBAU")
    print("-"*80)
    sets = build_index_sets(data)
    
    # Schritt 3: Parameter extrahieren
    print("\n" + "-"*80)
    print("SCHRITT 3: PARAMETER-EXTRAKTION")
    print("-"*80)
    params = extract_parameters(data, sets)
    
    # Schritt 4: Modell aufbauen
    print("\n" + "-"*80)
    print("SCHRITT 4: MODELL-FORMULIERUNG")
    print("-"*80)
    model, variables = build_model(data, sets, params)
    

    # ==============================
    # WARMSTART: Alle Diesel
    # ==============================

    sol = model.createSol()

    x = variables['x']
    y = variables['y']

    # 1) Alle Diesel-Fahrzeuge aktivieren (so viele wie Routen)
    for v in sets['V_D'][:len(sets['R'])]:
        model.setSolVal(sol, x[v], 1.0)

    # 2) Jede Route einem Diesel zuordnen
    for r_idx, r in enumerate(sets['R']):
        v = sets['V_D'][r_idx]
        model.setSolVal(sol, y[v, r], 1.0)

    # 3) Lösung an SCIP übergeben
    model.addSol(sol)

    # Schritt 5: Optimieren und Analysieren
    print("\n" + "-"*80)
    print("SCHRITT 5: OPTIMIERUNG & ANALYSE")
    print("-"*80)
    optimize_and_analyze(model, variables, sets, params)
    
    print("\n" + "="*80)
    print(" "*25 + "OPTIMIERUNG ABGESCHLOSSEN")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
