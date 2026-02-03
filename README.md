# README: Lineare Optimierungsprobleme mit PySCIPOpt

## 1. Allgemeine Grundlagen zur Vorlesung

### 1.1 Was ist ein lineares Optimierungsproblem?

Ein **lineares Optimierungsproblem** (auch: Linear Programming Problem, LP) ist eine mathematische Problemstellung, bei der wir:

- **eine lineare Zielfunktion** minimieren oder maximieren möchten
- **unter linearen Nebenbedingungen** (Restriktionen) und
- mit **Entscheidungsvariablen**, die kontinuierlich oder binär sein können

**Praktisches Verständnis:** Stellen Sie sich vor, ein Unternehmen möchte die **Gesamtkosten minimieren** oder **Gewinne maximieren**. Gleichzeitig muss es aber Einschränkungen beachten wie:
- Verfügbare Kapazitäten von Fabriken
- Kundennachfrage, die bedient werden muss
- Budgets und Ressourcen

Die Lösung des Optimierungsproblems sagt uns: **Welche Entscheidungen treffen wir am besten, um das Ziel zu erreichen?**

**Mathematisch ausgedrückt:**

```
min c₁x₁ + c₂x₂ + ... + cₙxₙ
Unter den Nebenbedingungen (constraints):
a₁₁x₁ + a₁₂x₂ + ... ≤ b₁
a₂₁x₁ + a₂₂x₂ + ... ≤ b₂
...
xᵢ ≥ 0 (Nichtnegativitätsbedingung)
```

---

### 1.2 Lösung mit Software – Unser Setting

#### Warum Software nutzen?

Optimierungsprobleme mit Hand zu lösen ist:
- **zeitaufwändig** (hunderte oder tausende Variablen)
- **fehleranfällig** (manuelle Rechenfehler)
- **unpraktisch** (Simplex-Algorithmus ist komplex)

Daher verwenden wir **Optimierungssoftware**, die automatisch die beste Lösung findet.

#### Unser Toolstack

Die Komponenten unseres Systems:

1. **SCIP (Solving Constraint Integer Programs)**
   - Der eigentliche **Solver** – das „Hirn" der Optimierung
   - Verwendet fortgeschrittene Algorithmen (Branch-and-Cut, Branch-and-Bound)
   - Findet die optimale oder nahezu optimale Lösung
   - Open Source und hochperformant

2. **PySCIPOpt (Python Wrapper für SCIP)**
   - Eine **Python-Schnittstelle** zu SCIP
   - Erlaubt uns, Optimierungsprobleme in Python zu formulieren
   - Übersetzt unsere Python-Befehle in SCIP-Befehle
   - Macht komplexe Optimierung „einfach"

3. **Python API**
   - Die **Programmiersprache** für unsere Implementierung
   - Wir definieren: Variablen, Parameter, Nebenbedingungen, Zielfunktion
   - Rufen dann den Solver auf, um die Lösung zu finden

4. **VSC (Visual Studio Code)**
   - Unsere **Entwicklungsumgebung** auf dem lokalen Computer
   - Hier schreiben, testen und debuggen wir den Code
   - **Warum nicht Google Colab?**
     - Colab läuft im **Browser** → Abhängig von Internetverbindung
     - Colab hat **begrenzte Rechenkraft** für größere Probleme
     - VSC läuft **lokal** → Volle Kontrolle, keine Abhängigkeiten
     - VSC ist **professioneller** für ernsthafte Entwicklung
     - Code bleibt **auf unserem Computer** (Datensicherheit)

---

## 2. Unsere Vorgehensweise

Wir folgen einem **systematischen Ansatz**, den wir aus der Vorlesung gelernt haben. Dabei orientieren wir uns an drei Fallstudien:

1. **Raffinerieproblem** (Rohöle und Kraftstoffe)
2. **Auswahlproblem Spediteur** (Fahrzeugauswahl)
3. **Fallstudie Juicy AG** (Produktionsplanung)

In jedem Fall gehen wir **in dieser standardisierten Reihenfolge** vor:

### 2.1 Schritt 1: Indexmengen definieren

**Was sind Indexmengen?**

Indexmengen sind endliche Mengen von Objekten, über die wir „zählen" oder summieren. Sie strukturieren das Problem und ermöglichen kompakte mathematische Formeln.

**Beispiele aus unserem Kontext:**
- $I$ = Menge der Produktionsstandorte (Fabriken)
- $J$ = Menge der Märkte (Absatzorte)
- $A$ = Menge der Ausbaustufen (Kapazitätserweiterungen)

**Warum brauchen wir sie?**
- Sie schaffen **Struktur** im Problem
- Sie ermöglichen **Verallgemeinerung** (statt „Fabrik 1, Fabrik 2, ..." schreiben wir einfach $i \in I$)
- Sie machen Formeln **übersichtlich** und **mathematisch präzise**

**Notation:**
Wenn wir schreiben $i \in I$, bedeutet das: „für alle Produktionsstandorte $i$ in der Menge $I$"

---

### 2.2 Schritt 2: Parameter definieren

**Was sind Parameter?**

Parameter sind gegebene Daten (Input), die wir **nicht selbst entscheiden** dürfen. Sie beschreiben die Rahmenbedingungen unseres Problems.

**Beispiele aus unserem Kontext:**

| Parameter | Notation | Bedeutung |
|-----------|----------|-----------|
| Nachfrage | $d_j$ | Wie viel muss für Markt $j$ produziert werden? |
| Variable Stückkosten | $cv_{ij}$ | Was kostet es, eine Einheit in Werk $i$ zu produzieren? |
| Kapazität | $cap_a^i$ | Wie viel kann Werk $i$ mit Ausbaustufe $a$ produzieren? |
| Fixkosten | $cf_a^i$ | Wie viel kostet es, Ausbaustufe $a$ in Werk $i$ zu bauen? |

**Praktisch gesprochen:** Diese Daten bekommen wir vom Management, von Lieferanten, aus Kostenbudgets – wir können sie nicht ändern.

---

### 2.3 Schritt 3: Entscheidungsvariablen definieren

**Was sind Entscheidungsvariablen?**

Entscheidungsvariablen (auch: Stellschrauben) sind Größen, die das Optimierungsmodell **selbst wählen soll**. Sie sind die „Hebel", über die wir die Zielfunktion optimieren.

**Beispiele aus unserem Kontext:**

| Variable | Notation | Typ | Bedeutung |
|----------|----------|-----|-----------|
| Produktions-/Transportmenge | $x_{ij} \geq 0$ | Kontinuierlich | Wie viel produzieren wir in Werk $i$ für Markt $j$? |
| Ausbau-Ja/Nein | $y_a^i \in \{0,1\}$ | Binär | Bauen wir Ausbaustufe $a$ in Werk $i$? (1=Ja, 0=Nein) |

**Der Unterschied:**
- $x_{ij}$ kann **jeden Wert annehmen** (0, 10, 100.5, ...)
- $y_a^i$ kann **nur 0 oder 1 sein** (Entweder-Oder-Entscheidung)

**Das Optimierungsproblem findet automatisch:** Welche Werte für $x_{ij}$ und $y_a^i$ minimieren unsere Gesamtkosten?

---

### 2.4 Schritt 4: Zielfunktion formulieren

**Was ist eine Zielfunktion?**

Die Zielfunktion ist das **Ziel**, das wir optimieren möchten. Sie fasst zusammen, was wir minimieren oder maximieren wollen.

#### Unsere Zielfunktion im Detail

$$\min \text{GK} = \sum_{i \in I} \sum_{j \in J} cv_{ij} \cdot x_{ij} + \sum_{i \in I} \sum_{a \in A} cf_a^i \cdot y_a^i$$

**Erklärung der Zielfunktion:**

**Teil (1): Operative Kosten**

$$\sum_{i \in I} \sum_{j \in J} cv_{ij} \cdot x_{ij}$$

- **Was wird hier berechnet?** Die Gesamtkosten für **Produktion und Transport**
- **Wie?** Für **jedes Werk $i$** und **jeden Markt $j$** multiplizieren wir:
  - $cv_{ij}$ (Kosten pro Einheit) mit
  - $x_{ij}$ (tatsächlich produzierte/transportierte Menge)
- **Dann summieren wir alles auf.**

**Beispiel:**
- Werk 1 → Markt A: 100 Einheiten × 5€/Einheit = 500€
- Werk 1 → Markt B: 50 Einheiten × 6€/Einheit = 300€
- Werk 2 → Markt A: 80 Einheiten × 4€/Einheit = 320€

**Teil (2): Investitionskosten**

$$\sum_{i \in I} \sum_{a \in A} cf_a^i \cdot y_a^i$$

- **Was wird hier berechnet?** Die Gesamtkosten für **Kapazitätserweiterungen** (Investitionen)
- **Wie?** Für **jedes Werk $i$** und **jede Ausbaustufe $a$**:
  - $cf_a^i$ (Fixkosten für den Ausbau, z.B. 1.000.000€) wird mit
  - $y_a^i$ (0 oder 1) multipliziert
  - Wenn $y_a^i = 1$ (Ausbau findet statt) → addieren wir $cf_a^i$
  - Wenn $y_a^i = 0$ (kein Ausbau) → addieren wir 0

**Gesamtzielfunktion:**

$$\text{Gesamtkosten} = \text{Operative Kosten} + \text{Investitionskosten}$$

Das Solver-Programm findet automatisch die **beste Kombination**.

---

### 2.5 Schritt 5: Restriktionen formulieren

**Was sind Restriktionen?**

Restriktionen sind **Einschränkungen** oder **Regeln**, die das Optimierungsproblem einhalten muss. Sie beschreiben die Realität: Was ist möglich, was nicht?

#### Restriktion (1): Nachfrage bedienen

$$\sum_{i \in I} x_{ij} = d_j \quad \forall j \in J$$

**Was bedeutet das?**
- **Linke Seite:** $\sum_{i \in I} x_{ij}$ = "Summe über alle Werke: Was wird insgesamt zu Markt $j$ geschickt?"
- **Rechte Seite:** $d_j$ = "Nachfrage von Markt $j$"
- **Bedingung:** Sie müssen **gleich sein!**

**Praktisch:**
Wenn Markt A 100 Einheiten braucht ($d_A = 100$), dann muss die Summe aller Lieferungen von allen Werken zu Markt A **genau 100 Einheiten** betragen.

**Warum wichtig?** Diese Restriktion garantiert, dass **alle Kunden beliefert werden** – eine geschäftliche Notwendigkeit.

#### Restriktion (2): Kapazitätsbedingung

$$\sum_{j \in J} x_{ij} \leq \sum_{a \in A} cap_a^i \cdot y_a^i \quad \forall i \in I$$

**Was bedeutet das?**
- **Linke Seite:** $\sum_{j \in J} x_{ij}$ = "Gesamtproduktion von Werk $i$ (für alle Märkte)"
- **Rechte Seite:** $\sum_{a \in A} cap_a^i \cdot y_a^i$ = "Verfügbare Kapazität in Werk $i$"
- **Bedingung:** Produktion $\leq$ Kapazität (kann nicht mehr produzieren als möglich!)

**Praktisch:**
Werk 1 kann mit den aktuell verbauten Maschinen (Ausbaustufe) maximal 500 Einheiten pro Tag produzieren.

**Warum wichtig?** Diese Restriktion garantiert **technische Machbarkeit**.

---

## 3. Kostentreiber, Engpässe und zentrale Einflussfaktoren

### 3.1 Was sind Kostentreiber?

**Definition:** Kostentreiber sind die **Hauptfaktoren**, die die Gesamtkosten des Unternehmens bestimmen.

**Kostentreiber in unserem Modell:**

| Kostentreiber | Parameter | Auswirkung |
|---------------|-----------|-----------|
| **Variable Produktionskosten** | $cv_{ij}$ | Je höher $cv_{ij}$, desto teurer die Produktion/der Transport |
| **Investitionskosten** | $cf_a^i$ | Große Investitionen für Ausbau können erheblich sein |
| **Nachfrage-Mix** | $d_j$ | Unterschiedliche Märkte kosten unterschiedlich zu bedienen |

---

### 3.2 Was sind Engpässe?

**Definition:** Engpässe (bottlenecks) sind **Ressourcen oder Kapazitäten**, die die Produktion limitieren.

**Engpässe in unserem Modell:**

| Engpass | Beschreibung | Problem | Lösung |
|---------|-------------|---------|--------|
| **Produktionskapazität** | Werk $i$ kann max. $\sum_{a} cap_a^i \cdot y_a^i$ Einheiten produzieren | Wenn Nachfrage > Kapazität | Werk ausbauen |
| **Transportkapazität** | Transportkorridore können überlastet sein | Transport wird zum Engpass | Alternative Routen suchen |
| **Raw Materials** | Verfügbarkeit von Rohstoffen begrenzt | Nicht unbegrenzt verfügbar | Mit begrenzten Ressourcen planen |

---

### 3.3 Was sind zentrale Einflussfaktoren?

**Definition:** Zentrale Einflussfaktoren sind **Variablen oder Parameter**, deren Änderung **signifikante Auswirkungen** auf die optimale Lösung hat.

**Zentrale Einflussfaktoren:**

| Einflussfaktor | Art | Auswirkung |
|----------------|-----|-----------|
| **Nachfrage $d_j$** | Parameter | Höhere Nachfrage → mehr produzieren → höhere Kosten |
| **Variable Kosten $cv_{ij}$** | Parameter | Höhere Kosten → weniger rentabel |
| **Kapazität $cap_a^i$** | Parameter | Größere Kapazität → mehr Spielraum |
| **Investitionskosten $cf_a^i$** | Parameter | Höhere Investitionskosten → schwerer zu rechtfertigen |
| **Ausbau-Entscheidung $y_a^i$** | Variable | Bestimmt langfristige Struktur |
| **Produktions-Mix $x_{ij}$** | Variable | Wie verteilen wir Produktion auf Werke? |

---

## 4. Zusammenhang: Modellstruktur verstehen

### 4.1 Der Modell-Aufbau schematisch

1. **INPUT** (Parameter & Indexmengen)
   - Indexmengen (I, J, A) → "Über was summieren wir?"
   - Parameter (d_j, cv_ij, cap_a^i, cf_a^i) → "Was sind die gegebenen Daten?"

2. **ENTSCHEIDUNGSVARIABLEN** (x_ij, y_a^i) → "Was soll das Modell entscheiden?"

3. **ZIELFUNKTION** → "Welches Ziel minimieren/maximieren?"

4. **RESTRIKTIONEN** → "Welche Regeln müssen eingehalten werden?"

5. **SOLVER** (SCIP über PySCIPOpt)

6. **OUTPUT**
   - Optimale Werte für x_ij und y_a^i
   - Minimale Gesamtkosten
   - Welche Werke ausbauen?
   - Welche Produktionsmengen und Routen?

---

## 5. Zusammenfassung für die Präsentation

### Die drei Säulen unserer Arbeit:

1. **Modellierung:** Wir übersetzen ein Business-Problem in mathematische Sprache
   - Indexmengen, Parameter, Variablen, Zielfunktion, Restriktionen

2. **Implementierung:** Wir verwenden PySCIPOpt + SCIP
   - Schreiben Python-Code in VSC
   - Der Solver findet die optimale Lösung automatisch

3. **Analyse:** Wir interpretieren die Ergebnisse
   - Welche Werke sollen ausgebaut werden?
   - Wie sollen Produktion und Transport optimal verteilt werden?
   - Was sind die Kostentreiber?
   - Wo sind Engpässe?

### Die Standard-Vorgehensweise (immer gleich):

1. **Indexmengen definieren** → strukturiert das Problem
2. **Parameter definieren** → beschreibt die Rahmenbedingungen
3. **Entscheidungsvariablen definieren** → beschreibt die Wahlmöglichkeiten
4. **Zielfunktion aufstellen** → beschreibt das Ziel
5. **Restriktionen formulieren** → beschreibt die Grenzen

Diese Struktur wird bei **jedem Problem** verwendet – ob Raffinerien, Spediteure oder Juicy AG!

---

## 6. Glossar – Wichtige Begriffe

| Begriff | Erklärung |
|---------|-----------|
| **Indexmenge** | Endliche Menge (z.B. {1, 2, 3, ...}), über die wir summieren |
| **Parameter** | Gegebene Daten, die wir nicht entscheiden können |
| **Entscheidungsvariable** | Größe, die das Optimierungsproblem selbst wählt |
| **Zielfunktion** | Das Ziel, das wir minimieren oder maximieren |
| **Restriktion** | Einschränkung/Regel, die die Lösung erfüllen muss |
| **Kostentreiber** | Faktoren, die die Gesamtkosten bestimmen |
| **Engpass** | Ressource/Kapazität, die die Produktion limitiert |
| **SCIP** | Solver-Engine für Optimierungsprobleme |
| **PySCIPOpt** | Python-Interface zu SCIP |
| **VSC** | Visual Studio Code – lokale Entwicklungsumgebung |
| **Kontinuierliche Variable** | Kann jeden Wert annehmen (z.B. 10.5, 100.3) |
| **Binäre Variable** | Kann nur 0 oder 1 sein |
| **Sensitivitätsanalyse** | Testen: "Was wenn dieser Parameter sich ändert?" |

---

## 7. Fragen, die Sie erwarten sollten (20-Min Q&A vorbereiten)

### Grundlagen (einfach)
- "Was ist ein Optimierungsproblem?" → Ziel minimieren unter Constraints
- "Warum brauchen wir Indexmengen?" → Struktur und kompakte Formeln
- "Was ist der Unterschied zwischen Parametern und Variablen?" → Parameter gegeben, Variablen zu wählen
- "Warum VSC und nicht Colab?" → Lokal, schneller, professioneller, Datensicherheit

### Modellierung (mittel)
- "Erklär mal die Nachfrage-Restriktion!" → Summe von Werk-zu-Markt muss Nachfrage erfüllen
- "Was passiert, wenn ein Engpass aktiv ist?" → Zusätzliche Kapazität wäre wertvoll
- "Wie sehen operative vs. Investitionskosten aus?" → Operative täglich, Investment einmalig
- "Warum zwei Arten von Variablen?" → Realistisch: täglich Mengen, aber selten Ausbau

### Lösung und Tools (mittel-schwer)
- "Wie findet SCIP die optimale Lösung?" → Branch-and-Bound, Branch-and-Cut Algorithmen
- "Was macht PySCIPOpt genau?" → Übersetzt unsere Python-Befehle in SCIP-Format
- "Wie lange dauert es, die Lösung zu finden?" → Abhängig von Problem-Größe

### Einsicht (schwer)
- "Welche sind die Hauptkostentreiber?" → $cv_{ij}$ und möglicherweise $cf_a^i$
- "Was sind die Engpässe?" → Kapazitäten der Werke
- "Was würden Sie ändern, wenn Transportkosten steigen?" → Möglicherweise dezentrale Produktion
- "Wann lohnt sich ein Ausbau?" → Wenn reduzierte operative Kosten Investitionskosten überkompensieren

---

**Diese Anleitung ist dein Fundament für die 20-Minuten-Präsentation und Q&A!**
