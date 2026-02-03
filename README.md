# README: Lineare Optimierungsprobleme mit PySCIPOpt

## 1. Allgemeine Grundlagen zur Vorlesung

### 1.1 Was ist ein lineares Optimierungsproblem?

Ein **lineares Optimierungsproblem** (auch: Linear Programming Problem, LP bzw. MILP bei binären Variablen) beschreibt eine mathematische Entscheidungsaufgabe, bei der

* eine **lineare Zielfunktion** minimiert wird,
* **lineare Nebenbedingungen (Restriktionen)** einzuhalten sind und
* **Entscheidungsvariablen** kontinuierlich und/oder ganzzahlig (insbesondere binär) sind.

In unserem Projekt wird ein **gemischt-ganzzahliges lineares Optimierungsproblem (MILP)** formuliert, da sowohl kontinuierliche Größen (Leistungen, Energiemengen, SOC) als auch binäre und ganzzahlige Entscheidungen (Fahrzeugbeschaffung, Tourenzuordnung, Infrastruktur) enthalten sind.

---

### 1.2 Lösung mit Software – Unser Setting

Die Lösung des Modells erfolgt vollständig softwaregestützt.

**Solver-Stack:**

* **SCIP**: mathematischer MILP-Solver (Branch-and-Bound / Branch-and-Cut)
* **PySCIPOpt**: Python-Schnittstelle zur Modellformulierung
* **Python**: Implementierung von Datenimport, Modellaufbau, Lösung und Auswertung
* **Visual Studio Code**: lokale Entwicklungsumgebung

Der vollständige Modellaufbau und die Optimierung sind in der Datei **`tk.py`** implementiert.

---

## 2. Unsere Vorgehensweise

Die Modellierung folgt exakt der in der Vorlesung vermittelten Standardstruktur:

1. Indexmengen
2. Parameter
3. Entscheidungsvariablen
4. Zielfunktion
5. Restriktionen

Alle folgenden Abschnitte beziehen sich **direkt auf die Implementierung in `tk.py`**.

---

## 2.1 Schritt 1: Indexmengen definieren

### Definition und Zweck

Indexmengen strukturieren das Modell und legen fest, **über welche Objekte summiert und indiziert wird**. Sie werden im Code zentral in der Funktion

> `build_index_sets(data)`

angelegt (Abschnitt *INDEXMENGEN AUFBAU* in `tk.py`).

### Im Modell verwendete Indexmengen

Alle Indexmengen werden im Folgenden in **LaTeX-Notation** angegeben.

* `R` – Menge aller Touren  
  Implementierung: `sets['R']`  
  Quelle: `routes.csv`  
  Bedeutung: Jede Tour muss genau einem Fahrzeug zugeordnet werden.

* `M_E` – Menge aller E‑Lkw-Modelle  
  Implementierung: `sets['M_E']`

* `M_D` – Menge aller Diesel‑Lkw-Modelle  
  Implementierung: `sets['M_D']`

* `V_E` – Menge aller E‑Lkw‑Instanzen  
  Implementierung: `sets['V_E']`  
  Erläuterung: Für jedes Modell werden mehrere potenzielle Fahrzeuginstanzen erzeugt, um ausreichend Freiheitsgrade für die Tourenzuordnung zu haben.

* `V_D` – Menge aller Diesel‑Lkw‑Instanzen  
  Implementierung: `sets['V_D']`

* `V = V_E \cup V_D` – Menge aller Fahrzeuge  
  Implementierung: `sets['V']`

* `T` – Menge der Zeitschritte eines Tages  
  Implementierung: `sets['T']`  
  Bedeutung: Diskretisierung des Tages in 96 Zeitschritte à 15 Minuten.

* `C` – Menge der Ladesäulentypen  
  Implementierung: `sets['C']`

* `S` – Menge der Ladepunkte (abgeleitet je Ladesäulentyp)  
  Implementierung: implizit über Ladepunktzuordnung in den Restriktionen

**Warum sind diese Indexmengen notwendig?**

Sie ermöglichen:

* kompakte mathematische Formulierungen,
* eine skalierbare Modellstruktur,
* eine 1‑zu‑1‑Abbildung zwischen mathematischem Modell und Code.

---

## 2.2 Schritt 2: Parameter definieren

### Definition

Parameter sind **exogen vorgegebene Daten**, die nicht optimiert werden. Sie werden aus CSV-Dateien oder aus der Konfigurationsklasse gelesen.

### Zentrale Parametergruppen im Code

Parameter werden in zwei Schritten definiert:

* **globale Parameter**: Klasse `OptimizationConfig`
* **instanzspezifische Parameter**: Funktion `extract_parameters(data, sets)`

### Wichtige Modellparameter (LaTeX-Notation)

* Zeitdiskretisierung:  
  `\Delta t` – Länge eines Zeitschritts

* Tourenparameter:  
  `d_r` – Gesamtdistanz von Tour `r`  
  `d_r^{toll}` – mautpflichtige Distanz von Tour `r`  
  `t_r^{start},\; t_r^{end}` – Start- und Endzeit einer Tour

* E‑Lkw‑Parameter:  
  `CAPEX_m^E,\; OPEX_m^E` – jährliche Fixkosten  
  `\varepsilon_m` – Energieverbrauch [kWh/km]  
  `Q_m` – Batteriekapazität  
  `P_m^{charge,max}` – maximale Ladeleistung  
  `THG_m` – THG‑Quotenerlös

* Diesel‑Parameter:  
  `CAPEX_m^D,\; OPEX_m^D,\; TAX_m^D`  
  `\kappa_m` – Dieselverbrauch [L/km]

* Infrastrukturparameter:  
  `CAPEX_c^{charge},\; OPEX_c^{charge}`  
  `P_c^{max}` – maximale Ladeleistung  
  `n_c^{spots}` – Ladepunkte je Säule

* Netz- und Speicherparameter:  
  `P_{grid}^{base},\; P_{grid}^{upgrade}`  
  `CAPEX_{stor}^P,\; CAPEX_{stor}^E`  
  `\eta_{charge},\; \eta_{discharge}`  
  `DoD_{min}`

Diese Parameter definieren vollständig den **technischen, wirtschaftlichen und zeitlichen Rahmen** des Modells.

---

## 2.3 Schritt 3: Entscheidungsvariablen definieren

### Definition

Entscheidungsvariablen sind die Größen, die der Solver wählt, um die Zielfunktion zu minimieren.

### Im Modell verwendete Variablen

#### Binäre und ganzzahlige Variablen

* `x_v \in \{0,1\}`  
  Fahrzeugbeschaffung (für `v \in V`)

* `y_{v,r} \in \{0,1\}`  
  Tourenzuordnung (Fahrzeug `v` fährt Tour `r`)

* `z_c \in \mathbb{Z}_+`  
  Anzahl installierter Ladesäulen vom Typ `c`

* `w_{v,s,t} \in \{0,1\}`  
  Ladebelegung eines Fahrzeugs an Ladepunkt `s` zur Zeit `t`

* `u_{grid} \in \{0,1\}`  
  Entscheidung über Netzerweiterung

#### Kontinuierliche Variablen

* `SOC_{v,t}` – Ladezustand E‑Lkw
* `p_{v,s,t}` – Ladeleistung
* `p_t^{grid}` – Netzbezugsleistung
* `p^{peak}` – Jahreshöchstlast
* `P^{stor},\; E^{stor}` – Speicherleistung und -kapazität
* `p_t^{stor,charge},\; p_t^{stor,discharge}` – Speicherleistungen
* `SOC_t^{stor}` – Speicher‑SOC

Alle Variablen werden in `build_model()` angelegt.

---

## 2.4 Schritt 4: Zielfunktion formulieren

### Struktur der Zielfunktion

Die Zielfunktion minimiert die **Total Cost of Ownership (TCO)** und unterscheidet explizit zwischen **operativen Kosten (OPEX)** und **Investitionskosten (CAPEX)**.

$`\min Z = Z_{vehicles}^E + Z_{vehicles}^D + Z_{infrastructure} + Z_{grid} + Z_{storage} + Z_{energy} - Z_{revenue}`$

### Kostenkomponenten

* **Fahrzeugkosten (CAPEX + OPEX)**  
  Leasing, Wartung und Steuern je Fahrzeuginstanz

* **Infrastrukturkosten (CAPEX + OPEX)**  
  Ladesäulen und Batteriespeicher

* **Netzkosten**  
  Grundgebühr, Netzerweiterung, Leistungspreis

* **Energiekosten (OPEX)**  
  Stromarbeitspreis, Diesel, Maut

* **Erlöse**  
  THG‑Quotenerlöse für E‑Lkw

Jede Kostenkomponente ist **direkt einer Variablen und einem Parameterblock zugeordnet**, sodass sich Änderungen transparent auf die optimale Lösung auswirken.

---

## 2.5 Schritt 5: Restriktionen formulieren

Im Modell werden **21 Restriktionen** implementiert. Sie bilden technische, zeitliche und logische Rahmenbedingungen ab.

### Übersicht aller Restriktionen

(1) Tourenzuordnung: jede Tour genau einmal  
(2) Tour nur mit beschafftem Fahrzeug  
(3) Keine zeitliche Überlappung von Touren  
(4) Maximale Anzahl installierter Ladesäulen  
(5) Ladepunktkapazität  
(6) Keine gleichzeitige Fahrt und Ladung  
(7) Linearisierung Ladeleistung (Big‑M)  
(8) Maximale Fahrzeugladeleistung  
(9) Leistungsgrenze je Ladesäulentyp  
(10) Depot‑Laderegeln (Nacht)  
(11) Batteriedynamik E‑Lkw  
(12) Zyklischer SOC E‑Lkw  
(13) SOC‑Grenzen E‑Lkw  
(14) Leistungsbilanz Depot  
(15) Netzanschlussgrenze  
(16) Peak‑Definition  
(17) Speicher‑Leistungsgrenzen  
(18) Speicher‑SOC‑Dynamik  
(19) Zyklischer Speicher‑SOC  
(20) Speicher‑SOC‑Grenzen mit DoD  
(21) Nicht‑Negativität

Alle Restriktionen sind im Abschnitt *NEBENBEDINGUNGEN* mathematisch dokumentiert und in `build_model()` implementiert.

---

## 3. Kostentreiber, Engpässe und Einflussfaktoren

### 3.1 Kostentreiber

Zentrale Kostentreiber im Modell sind:

* Fahrzeug‑CAPEX und ‑OPEX
* Strompreis und Peak‑Preis
* Diesel‑ und Mautkosten
* Ladeinfrastruktur‑Investitionen
* Batteriespeichergröße

Sie wirken **direkt additiv in der Zielfunktion** und bestimmen die optimale Flotten‑ und Infrastrukturstruktur.

---

### 3.2 Engpässe

* Ladeleistung und Anzahl Ladepunkte
* Netzanschlussleistung
* Zeitliche Überlappung von Touren
* Batteriekapazitäten

Aktive Engpässe führen zu höheren Grenzkosten und beeinflussen Ausbau‑ und Beschaffungsentscheidungen.

---

### 3.3 Zentrale Einflussfaktoren

* Tourenlängen und Zeitfenster
* Energiepreise
* Batterieparameter
* THG‑Erlöse
* Netzerweiterungskosten

Diese Parameter haben hohen Einfluss auf die optimale Lösung und eignen sich besonders für Sensitivitätsanalysen.

---

## 4. Zusammenfassung

Das Modell bildet ein **vollständiges, realitätsnahes MILP** zur Elektrifizierung eines Logistikdepots ab. Die klare Trennung von Indexmengen, Parametern, Variablen, Zielfunktion und Restriktionen erlaubt:

* saubere mathematische Interpretation,
* direkte Nachvollziehbarkeit im Code,
* fundierte ökonomische Analyse der Ergebnisse.

---

**Diese README erklärt das Modell vollständig auf Basis der Implementierung in `tk.py`.**
