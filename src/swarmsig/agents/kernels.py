"""Numba-Kerne für kleine Schwärme (Schnellpfad, Abschnitt 10.1), mit und ohne Wind (P1, P3).

Die Kerne rechnen dasselbe Modell wie die NumPy-Referenz (``bird.py``, ``drone.py``,
``realism.py``); die Gleichheit wird in ``tests/test_kernels.py`` geprüft. Die Nachbarsuche
läuft per Brute Force (O(N²)): für N ≲ 1000 schneller als der k-d-Baum, darüber nicht
empfohlen. Zufall kommt aus übergebenen ``numpy.random.Generator``-Objekten (Regel 4), je
Quelle ein eigener Strom (CRN-tauglich).

Wind (4.1, 4.9): u = v + w − c ŵ, ŵ ← ŵ + Δt/τ_est (w − ŵ). Der Wind w wird alle ``hold``
Schritte am alten Ort x^n zur neuen Zeit t_{n+1} ausgewertet und dazwischen gehalten; derselbe
Wert treibt Schätzer und Übertragung (Abweichung von 4.9, siehe ``dynamics/integrator.py``).

Parameter werden als Felder ``pf`` (float64) und ``pi`` (int64) übergeben; die Indizes
stehen in den Konstanten unten.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# --- Vögel ---------------------------------------------------------------------------------
(B_J, B_KAPPA, B_S0, B_USTAR, B_BETA, B_R0, B_RW, B_RHC, B_ETA, B_TAU0, B_RC, B_SIGMA, B_DT,
 B_KN, B_KZ, B_Z0, B_KH, B_ETX, B_ETY, B_XCX, B_XCY, B_CCOMP, B_TAUEST) = range(23)
N_BF = 23
BI_TOPO, BI_NC, BI_MARGINAL, BI_NAV, BI_NCOH, BI_WIND, BI_WHOLD = range(7)
N_BI = 7

# --- Drohnen -------------------------------------------------------------------------------
(D_S0, D_RC, D_SPACING, D_KAPPA, D_SIGMA, D_CA, D_CC, D_CS, D_A, D_B, D_H, D_EPS, D_C1, D_C2,
 D_RINT, D_PREP, D_R0F, D_CF, D_VF, D_PF, D_AF, D_TAUV, D_AMAX, D_VMAX, D_PLOSS, D_DT, D_SG,
 D_SGV, D_OUA, D_OUB, D_KN, D_KZ, D_Z0, D_KH, D_ETX, D_ETY, D_XCX, D_XCY, D_RX0, D_RY0, D_RZ0,
 D_RSPEED, D_RETX, D_RETY, D_RRAD, D_RPH0, D_RT0, D_CCOMP, D_TAUEST, D_WI, D_WS, D_KFORM) = range(52)
N_DF = 52
DI_KIND, DI_NCTRL, DI_LAGD, DI_MAXAGE, DI_KNN, DI_ATTR, DI_NAV, DI_REF, DI_WIND, DI_WHOLD, DI_OWN = range(11)
N_DI = 11
KIND_D1, KIND_D2, KIND_D2C, KIND_D3, KIND_DF = 1, 2, 3, 4, 5

# --- Wind (Parameterfeld ``wp`` von :meth:`TurbulentWind.params`) -----------------------------
W_U10, W_ALPHA, W_EX, W_EY, W_UC = range(5)
N_WP = 5


# ===========================================================================================
# Gemeinsame Hilfsfunktionen
# ===========================================================================================


@njit(cache=True)
def _select_knn(x, i, k, idx, dd):
    """Die k nächsten Nachbarn von i (ohne i), aufsteigend nach Abstand; Rückgabe: Anzahl."""
    n = x.shape[0]
    cnt = 0
    for j in range(n):
        if j == i:
            continue
        d2 = (x[j, 0] - x[i, 0]) ** 2 + (x[j, 1] - x[i, 1]) ** 2 + (x[j, 2] - x[i, 2]) ** 2
        if cnt < k or d2 < dd[k - 1]:
            p = cnt if cnt < k else k - 1
            while p > 0 and dd[p - 1] > d2:
                dd[p] = dd[p - 1]
                idx[p] = idx[p - 1]
                p -= 1
            dd[p] = d2
            idx[p] = j
            if cnt < k:
                cnt += 1
    return cnt


@njit(cache=True)
def _coh_f(r, r0, r_w, r_hc):
    f = np.tanh((r - r0) / r_w)
    if r < 2.0 * r_hc:
        rr = r if r > 0.5 * r_hc else 0.5 * r_hc
        f -= (r_hc / rr) ** 6 - 0.015625
    return f


@njit(cache=True)
def _nav(x, u, mode, kn, kz, z0, kh, etx, ety, xcx, xcy, nav):
    """Gleichförmiger Navigationsterm (4.2); mode 0 = Transit, 1 = Kreisen."""
    n = x.shape[0]
    mx = my = mz = ux = uy = 0.0
    for i in range(n):
        mx += x[i, 0]
        my += x[i, 1]
        mz += x[i, 2]
        ux += u[i, 0]
        uy += u[i, 1]
    mx /= n
    my /= n
    mz /= n
    ux /= n
    uy /= n
    if mode == 0:
        uh = np.sqrt(ux * ux + uy * uy)
        nav[0] = -kn * (ux - uh * etx)
        nav[1] = -kn * (uy - uh * ety)
    else:
        nav[0] = -kh * (mx - xcx)
        nav[1] = -kh * (my - xcy)
    nav[2] = -kz * (mz - z0)


@njit(cache=True)
def _sat(ax, ay, az, limit):
    nrm = np.sqrt(ax * ax + ay * ay + az * az)
    if nrm > limit:
        s = limit / nrm
        return ax * s, ay * s, az * s
    return ax, ay, az


@njit(cache=True)
def wind_eval(x, t, wp, mk, ma, mb, out):
    """Wind an den Positionen x zur Zeit t: Potenzprofil U10 (z/10 m)^α ê_U plus eingefrorene
    Kraichnan-Turbulenz Σ a_m cos(k_m·η) + b_m sin(k_m·η), η = x − U_c ê_U t (5.2, 5.3.4)."""
    n = x.shape[0]
    M = mk.shape[0]
    U10, alpha, ex, ey, Uc = wp[W_U10], wp[W_ALPHA], wp[W_EX], wp[W_EY], wp[W_UC]
    sx = Uc * ex * t
    sy = Uc * ey * t
    for i in range(n):
        z = x[i, 2]
        U = U10 * (z / 10.0) ** alpha if z > 0.0 else 0.0
        wx = U * ex
        wy = U * ey
        wz = 0.0
        px = x[i, 0] - sx
        py = x[i, 1] - sy
        pz = x[i, 2]
        for m in range(M):
            ph = mk[m, 0] * px + mk[m, 1] * py + mk[m, 2] * pz
            c = np.cos(ph)
            s = np.sin(ph)
            wx += ma[m, 0] * c + mb[m, 0] * s
            wy += ma[m, 1] * c + mb[m, 1] * s
            wz += ma[m, 2] * c + mb[m, 2] * s
        out[i, 0] = wx
        out[i, 1] = wy
        out[i, 2] = wz


@njit(cache=True)
def _advance(x, v, u, w_hat, w_cur, dt, wind_on, c_comp, tau_est):
    """u und x aus dem neuen v; mit Wind über die einheitliche Übertragung (4.1)."""
    n = x.shape[0]
    if wind_on:
        k_est = dt / tau_est
        for i in range(n):
            for c in range(3):
                w_hat[i, c] = w_hat[i, c] + k_est * (w_cur[i, c] - w_hat[i, c])
                u[i, c] = v[i, c] + w_cur[i, c] - c_comp * w_hat[i, c]
                x[i, c] = x[i, c] + u[i, c] * dt
    else:
        for i in range(n):
            for c in range(3):
                u[i, c] = v[i, c]
                x[i, c] = x[i, c] + u[i, c] * dt


# ===========================================================================================
# Vögel (B1, B3)
# ===========================================================================================


@njit(cache=True)
def bird_drift(x, u, v, pf, pi, rank, hist, head, count, nav, out):
    n = x.shape[0]
    topo = pi[BI_TOPO] == 1
    k = pi[BI_NC]
    marginal = pi[BI_MARGINAL] == 1
    J, kappa, s0, ustar = pf[B_J], pf[B_KAPPA], pf[B_S0], pf[B_USTAR]
    beta, r0, rw, rhc = pf[B_BETA], pf[B_R0], pf[B_RW], pf[B_RHC]
    eta, tau0, rc, dt = pf[B_ETA], pf[B_TAU0], pf[B_RC], pf[B_DT]
    L = hist.shape[0]
    kcoh = pi[BI_NCOH]
    nb = np.empty(n, np.int64)
    dd = np.empty(max(k, 1))
    nbc = np.empty(max(kcoh, 1), np.int64)
    ddc = np.empty(max(kcoh, 1))
    for i in range(n):
        cnt = 0
        if topo:
            cnt = _select_knn(x, i, k, nb, dd)
        else:
            for j in range(n):
                if j != i:
                    d2 = (x[j, 0] - x[i, 0]) ** 2 + (x[j, 1] - x[i, 1]) ** 2 + (x[j, 2] - x[i, 2]) ** 2
                    if np.sqrt(d2) < rc:
                        nb[cnt] = j
                        cnt += 1
        wsum = 0.0
        ax = ay = az = 0.0
        cx = cy = cz = 0.0
        ccnt = cnt
        if kcoh > 0:  # Zusammenhalt über eigene, größere topologische Nachbarschaft
            ccnt = _select_knn(x, i, kcoh, nbc, ddc)
            for m in range(ccnt):
                j = nbc[m]
                dx, dy, dz = x[j, 0] - x[i, 0], x[j, 1] - x[i, 1], x[j, 2] - x[i, 2]
                r = np.sqrt(dx * dx + dy * dy + dz * dz)
                if r > 0.0:
                    f = _coh_f(r, r0, rw, rhc)
                    cx += f * dx / r
                    cy += f * dy / r
                    cz += f * dz / r
        for m in range(cnt):
            j = nb[m]
            w = 1.0 if eta == 0.0 else np.exp(eta * (rank[j] - rank[i]))
            if tau0 > 0.0:
                lag = int(np.rint(tau0 * max(rank[j] - rank[i], 0.0) / dt))
                lag = min(lag, count - 1)
                row = (head - lag) % L
                ujx, ujy, ujz = hist[row, j, 0], hist[row, j, 1], hist[row, j, 2]
            else:
                ujx, ujy, ujz = u[j, 0], u[j, 1], u[j, 2]
            ax += w * (ujx - u[i, 0])
            ay += w * (ujy - u[i, 1])
            az += w * (ujz - u[i, 2])
            wsum += w
            if kcoh == 0:
                dx, dy, dz = x[j, 0] - x[i, 0], x[j, 1] - x[i, 1], x[j, 2] - x[i, 2]
                r = np.sqrt(dx * dx + dy * dy + dz * dz)
                if r > 0.0:
                    f = _coh_f(r, r0, rw, rhc)
                    cx += f * dx / r
                    cy += f * dy / r
                    cz += f * dz / r
        s = np.sqrt(v[i, 0] ** 2 + v[i, 1] ** 2 + v[i, 2] ** 2)
        dev = s - s0
        g = dev ** 3 / ustar ** 2 if marginal else dev
        sp = -kappa * g / s if s > 0.0 else 0.0
        ia = 1.0 / wsum if wsum > 0.0 else 0.0
        ic = 1.0 / ccnt if ccnt > 0 else 0.0
        out[i, 0] = J * ax * ia + sp * v[i, 0] + beta * cx * ic + nav[0]
        out[i, 1] = J * ay * ia + sp * v[i, 1] + beta * cy * ic + nav[1]
        out[i, 2] = J * az * ia + sp * v[i, 2] + beta * cz * ic + nav[2]


@njit(cache=True)
def bird_run(x, v, u, n_steps, step0, t0, pf, pi, rank, hist, hc, rng, wp, mk, ma, mb, w_hat, w_cur):
    """Integriert ``n_steps`` Euler–Maruyama-Schritte in place; ``hc`` = [head, count]."""
    n = x.shape[0]
    a = np.empty((n, 3))
    nav = np.zeros(3)
    dt = pf[B_DT]
    sq = pf[B_SIGMA] * np.sqrt(dt)
    L = hist.shape[0]
    use_hist = pf[B_TAU0] > 0.0
    wind_on = pi[BI_WIND] == 1
    hold = max(pi[BI_WHOLD], 1)
    for s in range(n_steps):
        step = step0 + s
        _nav(x, u, pi[BI_NAV], pf[B_KN], pf[B_KZ], pf[B_Z0], pf[B_KH], pf[B_ETX], pf[B_ETY],
             pf[B_XCX], pf[B_XCY], nav)
        bird_drift(x, u, v, pf, pi, rank, hist, hc[0], hc[1], nav, a)
        for i in range(n):
            for c in range(3):
                v[i, c] = (v[i, c] + a[i, c] * dt) + sq * rng.standard_normal()
        if wind_on and step % hold == 0:
            wind_eval(x, t0 + (step + 1) * dt, wp, mk, ma, mb, w_cur)
        _advance(x, v, u, w_hat, w_cur, dt, wind_on, pf[B_CCOMP], pf[B_TAUEST])
        if use_hist:
            hc[0] = (hc[0] + 1) % L
            for i in range(n):
                for c in range(3):
                    hist[hc[0], i, c] = u[i, c]
            hc[1] = min(hc[1] + 1, L)


# ===========================================================================================
# Drohnen (D1, D2, D2-c, D3) mit Realismus-Schicht
# ===========================================================================================


@njit(cache=True)
def _sigma_norm(z, eps):
    return (np.sqrt(1.0 + eps * z * z) - 1.0) / eps


@njit(cache=True)
def _rho_h(z, h):
    if 0.0 <= z < h:
        return 1.0
    if h <= z <= 1.0:
        return 0.5 * (1.0 + np.cos(np.pi * (z - h) / (1.0 - h)))
    return 0.0


@njit(cache=True)
def _phi(z, a, b):
    c = abs(a - b) / np.sqrt(4.0 * a * b)
    zz = z + c
    return 0.5 * ((a + b) * zz / np.sqrt(1.0 + zz * zz) + (a - b))


@njit(cache=True)
def _braking(r, a, p):
    if r < 0.0:
        return 0.0
    if r <= a / (p * p):
        return p * r
    return np.sqrt(max(2.0 * a * r - a * a / (p * p), 0.0))


@njit(cache=True)
def _reference(t, pf, mode):
    tau = t - pf[D_RT0]
    if mode == 0:
        sp = pf[D_RSPEED]
        return (pf[D_RX0] + sp * pf[D_RETX] * tau, pf[D_RY0] + sp * pf[D_RETY] * tau, pf[D_RZ0],
                sp * pf[D_RETX], sp * pf[D_RETY], 0.0)
    om = pf[D_RSPEED] / pf[D_RRAD]
    ph = pf[D_RPH0] + om * tau
    return (pf[D_RX0] + pf[D_RRAD] * np.cos(ph), pf[D_RY0] + pf[D_RRAD] * np.sin(ph), pf[D_RZ0],
            -pf[D_RSPEED] * np.sin(ph), pf[D_RSPEED] * np.cos(ph), 0.0)


@njit(cache=True)
def drone_command(x, u, v, t, pf, pi, hist_x, hist_u, head, count, age, rng_loss, cmd, ex, eu, offs):
    """Befehle aller Drohnen. ``ex``, ``eu``: eigene GPS-Fehler (N, 3), wirksam nur mit DI_OWN;
    ``offs``: feste Formationsversätze (nur DF)."""
    n = x.shape[0]
    kind = pi[DI_KIND]
    if pi[DI_OWN] == 1:  # eigener Zustand, wie der Regler ihn kennt (Graph und Navigation: wahr)
        xo = x + ex
        uo = u + eu
        vo = v + eu
    else:
        xo = x
        uo = u
        vo = v
    if kind == KIND_DF:  # Formation ohne Nachbarkopplung: kein Graph, keine Kommunikation
        r0, r1, r2, q0, q1, q2 = _reference(t, pf, pi[DI_REF])
        kf = pf[D_KFORM]
        for i in range(n):
            cx = q0 + kf * (r0 + offs[i, 0] - xo[i, 0])
            cy = q1 + kf * (r1 + offs[i, 1] - xo[i, 1])
            cz = q2 + kf * (r2 + offs[i, 2] - xo[i, 2])
            cmd[i, 0], cmd[i, 1], cmd[i, 2] = _sat(cx, cy, cz, pf[D_VMAX])
        return
    nctrl, lagd, maxage, knn = pi[DI_NCTRL], pi[DI_LAGD], pi[DI_MAXAGE], pi[DI_KNN]
    L = hist_x.shape[0]
    rng_max = pf[D_RINT] if (kind == KIND_D2 or kind == KIND_D2C) else pf[D_RC]
    nav = np.zeros(3)
    _nav(x, u, pi[DI_NAV], pf[D_KN], pf[D_KZ], pf[D_Z0], pf[D_KH], pf[D_ETX], pf[D_ETY],
         pf[D_XCX], pf[D_XCY], nav)
    xr0, xr1, xr2, ur0, ur1, ur2 = _reference(t, pf, pi[DI_REF])
    mx = my = mz = mux = muy = muz = 0.0
    for i in range(n):
        mx += x[i, 0]
        my += x[i, 1]
        mz += x[i, 2]
        mux += u[i, 0]
        muy += u[i, 1]
        muz += u[i, 2]
    mx /= n
    my /= n
    mz /= n
    mux /= n
    muy /= n
    muz /= n
    r_alpha = _sigma_norm(pf[D_RINT], pf[D_EPS])
    d_alpha = _sigma_norm(pf[D_SPACING], pf[D_EPS])
    tauv = pf[D_TAUV]
    nb = np.empty(n, np.int64)
    nbd = np.empty(n)
    inset = np.zeros(n, np.bool_)
    for i in range(n):
        # 1) Kommunikationsgraph mit wahren Abständen (aufsteigend nach j)
        cnt = 0
        for j in range(n):
            if j != i:
                r = np.sqrt((x[j, 0] - x[i, 0]) ** 2 + (x[j, 1] - x[i, 1]) ** 2 + (x[j, 2] - x[i, 2]) ** 2)
                if r < rng_max:
                    nb[cnt] = j
                    nbd[cnt] = r
                    cnt += 1
        if knn > 0 and cnt > knn:
            order = np.argsort(nbd[:cnt], kind="mergesort")[:knn]
            sel = np.sort(nb[:cnt][order])
            cnt = knn
            for m in range(cnt):
                nb[m] = sel[m]
        for j in range(n):
            inset[j] = False
        for m in range(cnt):
            inset[nb[m]] = True
        for j in range(n):
            if not inset[j]:
                age[i, j] = 0  # Kante existiert nicht mehr -> kein Paketgedächtnis
        # 2) Paketverlust, Latenz, Beobachtung und Reglerterme
        s_i = np.sqrt(vo[i, 0] ** 2 + vo[i, 1] ** 2 + vo[i, 2] ** 2)
        hx = hy = hz = 0.0
        if s_i > 0.0:
            hx, hy, hz = vo[i, 0] / s_i, vo[i, 1] / s_i, vo[i, 2] / s_i
        a1x = a1y = a1z = 0.0  # Ausrichtung / Konsens / Reibung
        a2x = a2y = a2z = 0.0  # Zusammenhalt / Gradient / Abstoßung
        a3x = a3y = a3z = 0.0  # Trennung (D1)
        for m in range(cnt):
            j = nb[m]
            if rng_loss.random() < pf[D_PLOSS]:
                age[i, j] = min(age[i, j] + nctrl, maxage)
            else:
                age[i, j] = 0
            lag = min(lagd + age[i, j], count - 1)
            row = (head - lag) % L
            dx = hist_x[row, j, 0] - xo[i, 0]
            dy = hist_x[row, j, 1] - xo[i, 1]
            dz = hist_x[row, j, 2] - xo[i, 2]
            dux = hist_u[row, j, 0] - uo[i, 0]
            duy = hist_u[row, j, 1] - uo[i, 1]
            duz = hist_u[row, j, 2] - uo[i, 2]
            r = np.sqrt(dx * dx + dy * dy + dz * dz)
            if kind == KIND_D1:
                a1x += dux
                a1y += duy
                a1z += duz
                a2x += dx
                a2y += dy
                a2z += dz
                if r < pf[D_SPACING] and r > 0.0:
                    a3x += dx / r ** 2
                    a3y += dy / r ** 2
                    a3z += dz / r ** 2
            elif kind == KIND_D2 or kind == KIND_D2C:
                z = _sigma_norm(r, pf[D_EPS])
                q = 1.0 / np.sqrt(1.0 + pf[D_EPS] * r * r)
                ph = _rho_h(z / r_alpha, pf[D_H]) * _phi(z - d_alpha, pf[D_A], pf[D_B])
                a2x += ph * dx * q
                a2y += ph * dy * q
                a2z += ph * dz * q
                aij = _rho_h(z / r_alpha, pf[D_H])
                a1x += aij * dux
                a1y += aij * duy
                a1z += aij * duz
            else:  # D3
                ex = ey = ez = 0.0
                if r > 0.0:
                    ex, ey, ez = dx / r, dy / r, dz / r
                if r < pf[D_SPACING]:
                    f = -pf[D_PREP] * (pf[D_SPACING] - r)
                    a2x += f * ex
                    a2y += f * ey
                    a2z += f * ez
                elif pi[DI_ATTR] == 1:
                    f = pf[D_PREP] * (r - pf[D_SPACING])
                    a2x += f * ex
                    a2y += f * ey
                    a2z += f * ez
                vij = np.sqrt(dux * dux + duy * duy + duz * duz)
                vmax_fr = max(pf[D_VF], _braking(r - pf[D_R0F], pf[D_AF], pf[D_PF]))
                if vij > vmax_fr:
                    f = pf[D_CF] * (vij - vmax_fr) / vij
                    a1x += f * dux
                    a1y += f * duy
                    a1z += f * duz
        if kind == KIND_D1:
            ic = 1.0 / cnt if cnt > 0 else 0.0
            sp = -pf[D_KAPPA] * (s_i - pf[D_S0])
            cmd[i, 0] = vo[i, 0] + tauv * (pf[D_CA] * a1x * ic + pf[D_CC] * a2x * ic - pf[D_CS] * a3x + sp * hx) + tauv * nav[0]
            cmd[i, 1] = vo[i, 1] + tauv * (pf[D_CA] * a1y * ic + pf[D_CC] * a2y * ic - pf[D_CS] * a3y + sp * hy) + tauv * nav[1]
            cmd[i, 2] = vo[i, 2] + tauv * (pf[D_CA] * a1z * ic + pf[D_CC] * a2z * ic - pf[D_CS] * a3z + sp * hz) + tauv * nav[2]
        elif kind == KIND_D2 or kind == KIND_D2C:
            if kind == KIND_D2C:
                nx = -pf[D_C1] * (mx - xr0) - pf[D_C2] * (mux - ur0)
                ny = -pf[D_C1] * (my - xr1) - pf[D_C2] * (muy - ur1)
                nz = -pf[D_C1] * (mz - xr2) - pf[D_C2] * (muz - ur2)
            else:
                nx = -pf[D_C1] * (xo[i, 0] - xr0) - pf[D_C2] * (uo[i, 0] - ur0)
                ny = -pf[D_C1] * (xo[i, 1] - xr1) - pf[D_C2] * (uo[i, 1] - ur1)
                nz = -pf[D_C1] * (xo[i, 2] - xr2) - pf[D_C2] * (uo[i, 2] - ur2)
            cmd[i, 0] = a2x + a1x + nx
            cmd[i, 1] = a2y + a1y + ny
            cmd[i, 2] = a2z + a1z + nz
        else:
            vx = pf[D_S0] * hx + a2x + a1x + tauv * nav[0]
            vy = pf[D_S0] * hy + a2y + a1y + tauv * nav[1]
            vz = pf[D_S0] * hz + a2z + a1z + tauv * nav[2]
            cmd[i, 0], cmd[i, 1], cmd[i, 2] = _sat(vx, vy, vz, pf[D_VMAX])


@njit(cache=True)
def drone_run(x, v, u, n_steps, step0, t0, pf, pi, cmd, has_cmd, hist_x, hist_u, hc, age,
              eps_x, eps_u, rng_noise, rng_gx, rng_gv, rng_loss, wp, mk, ma, mb, w_hat, w_cur,
              eps_xs, eps_us, rng_gxs, rng_gvs, offs):
    """Integriert ``n_steps`` Schritte inkl. Regeltakt, Kommunikation und GPS-Rauschen."""
    n = x.shape[0]
    dt = pf[D_DT]
    sq = pf[D_SIGMA] * np.sqrt(dt)
    kind = pi[DI_KIND]
    nctrl = pi[DI_NCTRL]
    L = hist_x.shape[0]
    oua, sgb, sgvb = pf[D_OUA], pf[D_SG] * pf[D_OUB], pf[D_SGV] * pf[D_OUB]
    wind_on = pi[DI_WIND] == 1
    hold = max(pi[DI_WHOLD], 1)
    own = pi[DI_OWN] == 1
    wi, ws = pf[D_WI], pf[D_WS]
    acc = np.empty((n, 3))
    ex = np.empty((n, 3))
    eu = np.empty((n, 3))
    for s in range(n_steps):
        step = step0 + s
        for i in range(n):  # aktuelle GPS-Fehler: unabhängiger plus gemeinsamer Anteil
            for c in range(3):
                ex[i, c] = wi * eps_x[i, c] + ws * eps_xs[c]
                eu[i, c] = wi * eps_u[i, c] + ws * eps_us[c]
        if has_cmd[0] == 0 or step % nctrl == 0:
            drone_command(x, u, v, t0 + step * dt, pf, pi, hist_x, hist_u, hc[0], hc[1], age,
                          rng_loss, cmd, ex, eu, offs)
            has_cmd[0] = 1
        for i in range(n):
            if kind == KIND_D2 or kind == KIND_D2C:
                ax, ay, az = cmd[i, 0], cmd[i, 1], cmd[i, 2]
            elif own:  # innerer Regelkreis auf der eigenen Geschwindigkeitsschätzung
                ax = (cmd[i, 0] - (v[i, 0] + eu[i, 0])) / pf[D_TAUV]
                ay = (cmd[i, 1] - (v[i, 1] + eu[i, 1])) / pf[D_TAUV]
                az = (cmd[i, 2] - (v[i, 2] + eu[i, 2])) / pf[D_TAUV]
            else:
                ax = (cmd[i, 0] - v[i, 0]) / pf[D_TAUV]
                ay = (cmd[i, 1] - v[i, 1]) / pf[D_TAUV]
                az = (cmd[i, 2] - v[i, 2]) / pf[D_TAUV]
            acc[i, 0], acc[i, 1], acc[i, 2] = _sat(ax, ay, az, pf[D_AMAX])
        for i in range(n):
            for c in range(3):
                v[i, c] = (v[i, c] + acc[i, c] * dt) + sq * rng_noise.standard_normal()
        for i in range(n):
            v[i, 0], v[i, 1], v[i, 2] = _sat(v[i, 0], v[i, 1], v[i, 2], pf[D_VMAX])
        if wind_on and step % hold == 0:
            wind_eval(x, t0 + (step + 1) * dt, wp, mk, ma, mb, w_cur)
        _advance(x, v, u, w_hat, w_cur, dt, wind_on, pf[D_CCOMP], pf[D_TAUEST])
        for i in range(n):
            for c in range(3):
                eps_x[i, c] = oua * eps_x[i, c] + sgb * rng_gx.standard_normal()
        for i in range(n):
            for c in range(3):
                eps_u[i, c] = oua * eps_u[i, c] + sgvb * rng_gv.standard_normal()
        for c in range(3):
            eps_xs[c] = oua * eps_xs[c] + sgb * rng_gxs.standard_normal()
        for c in range(3):
            eps_us[c] = oua * eps_us[c] + sgvb * rng_gvs.standard_normal()
        hc[0] = (hc[0] + 1) % L
        for i in range(n):
            for c in range(3):
                hist_x[hc[0], i, c] = x[i, c] + (wi * eps_x[i, c] + ws * eps_xs[c])
                hist_u[hc[0], i, c] = u[i, c] + (wi * eps_u[i, c] + ws * eps_us[c])
        hc[1] = min(hc[1] + 1, L)
