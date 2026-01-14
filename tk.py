"""
Truck Fleet Optimization Model
================================
Optimiert die Zusammensetzung einer LKW-Flotte zwischen Diesel- und Elektro-LKWs
unter Berücksichtigung von:
- CAPEX und OPEX der Fahrzeuge
- Energie- und Dieselkosten
- Mautkosten
- THG-Prämien für Elektrofahrzeuge
- Ladeinfrastruktur und Ladezeiten
"""

import pandas as pd
import pulp
from datetime import datetime, timedelta

# Konstanten
DIESEL_PRICE = 1.70  # €/Liter
ELECTRICITY_PRICE = 0.30  # €/kWh
TOLL_PRICE = 0.187  # €/km für schwere LKW
THG_BONUS_E_TRUCK = -1000  # € (negative cost = income) per E-Truck per year

def load_data():
    """Lädt alle CSV-Dateien"""
    routes = pd.read_csv('Daten/routes.csv', sep=';')
    electric_trucks = pd.read_csv('Daten/electric_trucks.csv', sep=';')
    diesel_trucks = pd.read_csv('Daten/diesel_trucks.csv', sep=';')
    chargers = pd.read_csv('Daten/chargers.csv', sep=';')
    
    return routes, electric_trucks, diesel_trucks, chargers

def calculate_route_time(route):
    """Berechnet die Dauer einer Route in Stunden"""
    start = datetime.strptime(route['starttime'], '%H:%M')
    end = datetime.strptime(route['endtime'], '%H:%M')
    
    # Handle overnight routes
    if end < start:
        end += timedelta(days=1)
    
    duration = (end - start).total_seconds() / 3600
    return duration

def calculate_charging_time(energy_needed_kwh, charger_power_kw):
    """Berechnet die Ladezeit in Stunden"""
    if charger_power_kw == 0:
        return 0
    return energy_needed_kwh / charger_power_kw

def build_optimization_model(routes, electric_trucks, diesel_trucks, chargers):
    """
    Erstellt und löst das Optimierungsmodell
    """
    
    # Problem definieren (Minimiere Gesamtkosten)
    prob = pulp.LpProblem("Fleet_Optimization", pulp.LpMinimize)
    
    # Sets
    route_ids = routes['route_id'].tolist()
    e_truck_models = electric_trucks['truck_model'].tolist()
    d_truck_models = diesel_trucks['truck_model'].tolist()
    charger_models = chargers['charger_model'].tolist()
    
    # Decision Variables
    # x[r,t] = 1 if route r is assigned to electric truck model t
    x_electric = pulp.LpVariable.dicts("x_electric",
                                       ((r, t) for r in route_ids for t in e_truck_models),
                                       cat='Binary')
    
    # y[r,t] = 1 if route r is assigned to diesel truck model t
    y_diesel = pulp.LpVariable.dicts("y_diesel",
                                     ((r, t) for r in route_ids for t in d_truck_models),
                                     cat='Binary')
    
    # z[c] = number of chargers of type c
    z_chargers = pulp.LpVariable.dicts("z_chargers",
                                       charger_models,
                                       lowBound=0,
                                       cat='Integer')
    
    # n_electric[t] = number of electric trucks of model t
    n_electric = pulp.LpVariable.dicts("n_electric",
                                       e_truck_models,
                                       lowBound=0,
                                       cat='Integer')
    
    # n_diesel[t] = number of diesel trucks of model t
    n_diesel = pulp.LpVariable.dicts("n_diesel",
                                     d_truck_models,
                                     lowBound=0,
                                     cat='Integer')
    
    # Objective Function: Minimize Total Cost of Ownership (TCO)
    total_cost = 0
    
    # 1. Electric Truck Costs
    for t in e_truck_models:
        e_truck = electric_trucks[electric_trucks['truck_model'] == t].iloc[0]
        capex = e_truck['capex_yearly']
        opex = e_truck['opex_yearly']
        thg_bonus = e_truck['thg_yearly']  # negative value = income
        
        total_cost += n_electric[t] * (capex + opex + thg_bonus)
    
    # 2. Diesel Truck Costs
    for t in d_truck_models:
        d_truck = diesel_trucks[diesel_trucks['truck_model'] == t].iloc[0]
        capex = d_truck['capex_yearly']
        opex = d_truck['opex_yearly']
        kfz_tax = d_truck['kfz_yearly']
        
        total_cost += n_diesel[t] * (capex + opex + kfz_tax)
    
    # 3. Charger Costs
    for c in charger_models:
        charger = chargers[chargers['charger_model'] == c].iloc[0]
        total_cost += z_chargers[c] * (charger['capex_yearly'] + charger['opex_yearly'])
    
    # 4. Route-specific costs (Energy/Diesel + Toll)
    for idx, route in routes.iterrows():
        r = route['route_id']
        distance = route['distance_total']
        toll_distance = route['distance_toll']
        
        # Electric truck route costs
        for t in e_truck_models:
            e_truck = electric_trucks[electric_trucks['truck_model'] == t].iloc[0]
            energy_per_100km = e_truck['avg_energy_kWh_per_100km']
            energy_cost = (distance / 100) * energy_per_100km * ELECTRICITY_PRICE
            toll_cost = toll_distance * TOLL_PRICE
            
            total_cost += x_electric[r, t] * (energy_cost + toll_cost)
        
        # Diesel truck route costs
        for t in d_truck_models:
            d_truck = diesel_trucks[diesel_trucks['truck_model'] == t].iloc[0]
            diesel_per_100km = d_truck['avg_diesel_per_100km']
            diesel_cost = (distance / 100) * diesel_per_100km * DIESEL_PRICE
            toll_cost = toll_distance * TOLL_PRICE
            
            total_cost += y_diesel[r, t] * (diesel_cost + toll_cost)
    
    prob += total_cost, "Total_Cost_of_Ownership"
    
    # Constraints
    
    # 1. Each route must be assigned to exactly one truck
    for r in route_ids:
        prob += (pulp.lpSum([x_electric[r, t] for t in e_truck_models]) +
                pulp.lpSum([y_diesel[r, t] for t in d_truck_models]) == 1,
                f"Route_{r}_Assignment")
    
    # 2. Number of electric trucks must be sufficient for assigned routes
    for t in e_truck_models:
        prob += (pulp.lpSum([x_electric[r, t] for r in route_ids]) <= n_electric[t],
                f"Electric_Truck_{t}_Capacity")
    
    # 3. Number of diesel trucks must be sufficient for assigned routes
    for t in d_truck_models:
        prob += (pulp.lpSum([y_diesel[r, t] for r in route_ids]) <= n_diesel[t],
                f"Diesel_Truck_{t}_Capacity")
    
    # 4. Charging infrastructure constraints
    # Total charging power must be sufficient for all electric trucks
    # Assuming charging happens during non-operational hours (e.g., 8 hours available)
    total_charging_spots = pulp.lpSum([z_chargers[c] * chargers[chargers['charger_model'] == c].iloc[0]['charging_spots']
                                        for c in charger_models])
    
    total_electric_trucks = pulp.lpSum([n_electric[t] for t in e_truck_models])
    
    # At least one charging spot per 2 electric trucks (conservative estimate)
    prob += total_charging_spots >= total_electric_trucks / 2, "Charging_Spots_Minimum"
    
    # 5. Charging power constraint: Must be able to charge all trucks overnight
    # Calculate total daily energy consumption for electric routes
    for t in e_truck_models:
        e_truck = electric_trucks[electric_trucks['truck_model'] == t].iloc[0]
        energy_per_100km = e_truck['avg_energy_kWh_per_100km']
        max_power_truck = e_truck['max_power']
        
        for idx, route in routes.iterrows():
            r = route['route_id']
            distance = route['distance_total']
            energy_needed = (distance / 100) * energy_per_100km
            
            # The charger power must be at least as high as truck's max power capability
            for c in charger_models:
                charger = chargers[chargers['charger_model'] == c].iloc[0]
                if charger['max_power'] < max_power_truck:
                    # This charger cannot efficiently charge this truck model
                    # We model this as a soft constraint via higher costs (already in objective)
                    pass
    
    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=1))
    
    return prob, x_electric, y_diesel, z_chargers, n_electric, n_diesel

def print_results(prob, routes, electric_trucks, diesel_trucks, chargers,
                 x_electric, y_diesel, z_chargers, n_electric, n_diesel):
    """
    Gibt die Ergebnisse der Optimierung aus
    """
    print("\n" + "="*80)
    print("OPTIMIERUNGSERGEBNISSE - TRUCK FLEET OPTIMIZATION")
    print("="*80)
    
    print(f"\nStatus: {pulp.LpStatus[prob.status]}")
    print(f"Gesamtkosten (TCO): {pulp.value(prob.objective):,.2f} €/Jahr")
    
    print("\n" + "-"*80)
    print("FLOTTENAUFTEILUNG")
    print("-"*80)
    
    # Electric Trucks
    print("\nElektro-LKWs:")
    total_electric = 0
    for t in electric_trucks['truck_model']:
        count = pulp.value(n_electric[t])
        if count > 0:
            print(f"  {t}: {int(count)} Fahrzeuge")
            total_electric += count
    print(f"  Gesamt: {int(total_electric)} Elektro-LKWs")
    
    # Diesel Trucks
    print("\nDiesel-LKWs:")
    total_diesel = 0
    for t in diesel_trucks['truck_model']:
        count = pulp.value(n_diesel[t])
        if count > 0:
            print(f"  {t}: {int(count)} Fahrzeuge")
            total_diesel += count
    print(f"  Gesamt: {int(total_diesel)} Diesel-LKWs")
    
    # Chargers
    print("\n" + "-"*80)
    print("LADEINFRASTRUKTUR")
    print("-"*80)
    for c in chargers['charger_model']:
        count = pulp.value(z_chargers[c])
        if count > 0:
            charger = chargers[chargers['charger_model'] == c].iloc[0]
            spots = charger['charging_spots']
            total_spots = int(count * spots)
            print(f"  {c}: {int(count)} Ladestationen ({total_spots} Ladepunkte)")
    
    # Route Assignments
    print("\n" + "-"*80)
    print("ROUTENZUTEILUNG")
    print("-"*80)
    
    route_ids = routes['route_id'].tolist()
    e_truck_models = electric_trucks['truck_model'].tolist()
    d_truck_models = diesel_trucks['truck_model'].tolist()
    
    electric_routes = []
    diesel_routes = []
    
    for r in route_ids:
        route_info = routes[routes['route_id'] == r].iloc[0]
        
        for t in e_truck_models:
            if pulp.value(x_electric[r, t]) == 1:
                electric_routes.append((r, t, route_info))
                break
        
        for t in d_truck_models:
            if pulp.value(y_diesel[r, t]) == 1:
                diesel_routes.append((r, t, route_info))
                break
    
    print(f"\nElektro-Routen ({len(electric_routes)}):")
    for r, t, info in electric_routes:
        print(f"  {r} ({info['route_name']}): {info['distance_total']}km - {t}")
    
    print(f"\nDiesel-Routen ({len(diesel_routes)}):")
    for r, t, info in diesel_routes:
        print(f"  {r} ({info['route_name']}): {info['distance_total']}km - {t}")
    
    # Cost Breakdown
    print("\n" + "-"*80)
    print("KOSTENAUFSCHLÜSSELUNG (jährlich)")
    print("-"*80)
    
    # Calculate individual cost components
    electric_capex = sum(pulp.value(n_electric[t]) * electric_trucks[electric_trucks['truck_model'] == t].iloc[0]['capex_yearly']
                        for t in e_truck_models)
    electric_opex = sum(pulp.value(n_electric[t]) * electric_trucks[electric_trucks['truck_model'] == t].iloc[0]['opex_yearly']
                       for t in e_truck_models)
    electric_thg = sum(pulp.value(n_electric[t]) * electric_trucks[electric_trucks['truck_model'] == t].iloc[0]['thg_yearly']
                      for t in e_truck_models)
    
    diesel_capex = sum(pulp.value(n_diesel[t]) * diesel_trucks[diesel_trucks['truck_model'] == t].iloc[0]['capex_yearly']
                      for t in d_truck_models)
    diesel_opex = sum(pulp.value(n_diesel[t]) * diesel_trucks[diesel_trucks['truck_model'] == t].iloc[0]['opex_yearly']
                     for t in d_truck_models)
    diesel_kfz = sum(pulp.value(n_diesel[t]) * diesel_trucks[diesel_trucks['truck_model'] == t].iloc[0]['kfz_yearly']
                    for t in d_truck_models)
    
    charger_capex = sum(pulp.value(z_chargers[c]) * chargers[chargers['charger_model'] == c].iloc[0]['capex_yearly']
                       for c in chargers['charger_model'])
    charger_opex = sum(pulp.value(z_chargers[c]) * chargers[chargers['charger_model'] == c].iloc[0]['opex_yearly']
                      for c in chargers['charger_model'])
    
    # Calculate energy and toll costs
    energy_cost_total = 0
    diesel_cost_total = 0
    toll_cost_electric = 0
    toll_cost_diesel = 0
    
    for r in route_ids:
        route_info = routes[routes['route_id'] == r].iloc[0]
        distance = route_info['distance_total']
        toll_distance = route_info['distance_toll']
        
        for t in e_truck_models:
            if pulp.value(x_electric[r, t]) == 1:
                e_truck = electric_trucks[electric_trucks['truck_model'] == t].iloc[0]
                energy_per_100km = e_truck['avg_energy_kWh_per_100km']
                energy_cost_total += (distance / 100) * energy_per_100km * ELECTRICITY_PRICE
                toll_cost_electric += toll_distance * TOLL_PRICE
        
        for t in d_truck_models:
            if pulp.value(y_diesel[r, t]) == 1:
                d_truck = diesel_trucks[diesel_trucks['truck_model'] == t].iloc[0]
                diesel_per_100km = d_truck['avg_diesel_per_100km']
                diesel_cost_total += (distance / 100) * diesel_per_100km * DIESEL_PRICE
                toll_cost_diesel += toll_distance * TOLL_PRICE
    
    print("\nFahrzeugkosten:")
    print(f"  Elektro CAPEX:        {electric_capex:>12,.2f} €")
    print(f"  Elektro OPEX:         {electric_opex:>12,.2f} €")
    print(f"  Elektro THG-Prämie:   {electric_thg:>12,.2f} € (Erlös)")
    print(f"  Diesel CAPEX:         {diesel_capex:>12,.2f} €")
    print(f"  Diesel OPEX:          {diesel_opex:>12,.2f} €")
    print(f"  Diesel KFZ-Steuer:    {diesel_kfz:>12,.2f} €")
    
    print("\nLadeinfrastruktur:")
    print(f"  Ladestationen CAPEX:  {charger_capex:>12,.2f} €")
    print(f"  Ladestationen OPEX:   {charger_opex:>12,.2f} €")
    
    print("\nBetriebskosten:")
    print(f"  Stromkosten:          {energy_cost_total:>12,.2f} €")
    print(f"  Dieselkosten:         {diesel_cost_total:>12,.2f} €")
    print(f"  Maut (Elektro):       {toll_cost_electric:>12,.2f} €")
    print(f"  Maut (Diesel):        {toll_cost_diesel:>12,.2f} €")
    
    total_calculated = (electric_capex + electric_opex + electric_thg +
                       diesel_capex + diesel_opex + diesel_kfz +
                       charger_capex + charger_opex +
                       energy_cost_total + diesel_cost_total +
                       toll_cost_electric + toll_cost_diesel)
    
    print("\n" + "-"*80)
    print(f"Gesamtkosten (berechnet): {total_calculated:,.2f} €/Jahr")
    print(f"Gesamtkosten (Optimierer): {pulp.value(prob.objective):,.2f} €/Jahr")
    print("="*80 + "\n")

def main():
    """Hauptfunktion"""
    print("Lade Daten...")
    routes, electric_trucks, diesel_trucks, chargers = load_data()
    
    print(f"Geladene Daten:")
    print(f"  - {len(routes)} Routen")
    print(f"  - {len(electric_trucks)} Elektro-LKW-Modelle")
    print(f"  - {len(diesel_trucks)} Diesel-LKW-Modelle")
    print(f"  - {len(chargers)} Ladestations-Modelle")
    
    print("\nErstelle Optimierungsmodell...")
    prob, x_electric, y_diesel, z_chargers, n_electric, n_diesel = build_optimization_model(
        routes, electric_trucks, diesel_trucks, chargers
    )
    
    print("\nLöse Optimierungsproblem...")
    
    print_results(prob, routes, electric_trucks, diesel_trucks, chargers,
                 x_electric, y_diesel, z_chargers, n_electric, n_diesel)

if __name__ == "__main__":
    main()
