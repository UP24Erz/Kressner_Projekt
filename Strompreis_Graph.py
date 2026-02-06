import math

def smoothstep(u: float) -> float:
    """C1-glatt, u in [0,1]."""
    return u*u*(3 - 2*u)

def lerp(a: float, b: float, w: float) -> float:
    return a + (b - a) * w

def strompreis_smooth(t: float, tau: float = 1.0) -> float:
    """
    Glatte Simulation der variablen Strompreise über 24h.
    t   : Uhrzeit in Stunden (float; beliebig, wird modulo 24 genommen)
    tau : Übergangsbreite in Stunden (z.B. 0.5 .. 2.0). Größer = weicher.
    Rückgabe: Preis in ct/kWh
    """
    # Preise je Segment
    p0, p1, p2, p3 = 0.18, 0.25, 0.35, 0.27

    # t in [0,24)
    t = t % 24.0
    h = tau / 2.0

    # Segment-Preis (ohne Glättung)
    if 0 <= t < 6:
        p = p0
    elif 6 <= t < 16:
        p = p1
    elif 16 <= t < 22:
        p = p2
    else:  # 22 <= t < 24
        p = p3

    # Hilfsfunktion: glatter Übergang um eine Grenze T von p_left -> p_right
    def blend_around_boundary(t_local: float, T: float, p_left: float, p_right: float) -> float | None:
        # Bereich [T-h, T+h]
        if (T - h) <= t_local <= (T + h):
            u = (t_local - (T - h)) / tau  # 0..1
            w = smoothstep(max(0.0, min(1.0, u)))
            return lerp(p_left, p_right, w)
        return None

    # Übergänge an 6,16,22
    for T, left, right in [
        (6.0,  p0, p1),
        (16.0, p1, p2),
        (22.0, p2, p3),
    ]:
        b = blend_around_boundary(t, T, left, right)
        if b is not None:
            return b

    # Übergang über Mitternacht: von p3 (22-24) zu p0 (0-6) um T=24 (=0)
    # Wir betrachten dazu t nahe 24 ODER nahe 0 als Bereich um 24.
    t_for_midnight = t if t >= (24.0 - h) else (t + 24.0)  # t in [24-h, 24+h]
    b = blend_around_boundary(t_for_midnight, 24.0, p3, p0)
    if b is not None:
        return b

    return p

if __name__ == "__main__":
    # Test / Demo
    for hour in range(0, 25):
        for minute in [0, 15, 30, 45]:
            t = hour + minute / 60.0
            p = strompreis_smooth(t, tau=1.0)
            print(f"t={t:5.2f} h -> p={p:.3f} €/kWh")