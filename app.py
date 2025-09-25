# 🛩️ eVTOL Mini-Lab — Streamlit demo
# Features: Planner + Perception + Health + Fleet UTM + Logs/Export + Physics Fidelity + Copilot (heuristic by default)
# Run: streamlit run app.py

import os, math, time, json
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="eVTOL Mini-Lab", page_icon="🛩️", layout="wide")
st.title("🛩️ eVTOL Mini-Lab")
st.caption("Autonomy • Perception • Predictive Maintenance • Fleet Coordination • Copilot")

# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
def astar(grid: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]):
    H, W = grid.shape
    def h(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    open_set = {start}; came_from={}; g={start:0}; f={start:h(start,goal)}
    while open_set:
        current = min(open_set, key=lambda n: f.get(n, 1e9))
        if current == goal:
            path=[current]
            while current in came_from:
                current=came_from[current]; path.append(current)
            return path[::-1]
        open_set.remove(current)
        for dy,dx in [(1,0),(-1,0),(0,1),(0,-1)]:
            ny, nx = current[0]+dy, current[1]+dx
            if not (0<=ny<H and 0<=nx<W): continue
            if grid[ny,nx]==1: continue
            tentative=g[current]+1
            if tentative < g.get((ny,nx),1e9):
                came_from[(ny,nx)]=current
                g[(ny,nx)]=tentative
                f[(ny,nx)]=tentative+h((ny,nx),goal)
                open_set.add((ny,nx))
    return None

def draw_grid(grid, start, goal, nfz_rects, scale_m=50, title="Grid"):
    fig, ax = plt.subplots(figsize=(5.5,5.5))
    ax.imshow(grid, cmap="Greys", origin="lower")
    for (y,x) in [start,goal]:
        if y >= 0 and x >= 0:
            ax.scatter(x,y,s=60,marker="o" if (y,x)==start else "X")
    for (y0,x0,y1,x1) in nfz_rects:
        rect=plt.Rectangle((x0,y0),x1-x0+1,y1-y0+1,fill=True,alpha=0.2)
        ax.add_patch(rect)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{title} (scale {scale_m} m/cell)")
    return fig, ax

def energy_wh(distance_m, cruise_ms, mass_kg, hover_s):
    # Simple model: hover ~ mass^1.5; cruise ~ V^3 drag
    k_hover=25.0
    P_hover=k_hover*(mass_kg**1.5)
    E_hover=P_hover*hover_s/3600.0
    P_cruise=(0.5*1.2*(cruise_ms**3)*0.6)/0.75
    t_cruise=distance_m/max(1e-6,cruise_ms)
    E_cruise=P_cruise*t_cruise/3600.0
    return E_hover+E_cruise, P_hover, P_cruise, t_cruise

# ▶ Normalized health score (0–100)
def compute_health(hours, cycles, max_temp_c, vib_g_rms):
    h_n = min(hours / 2000.0, 1.0)
    c_n = min(cycles / 2000.0, 1.0)
    t_excess = max(0.0, max_temp_c - 40.0)
    t_n = min(t_excess / 40.0, 1.0)
    v_n = min(vib_g_rms / 2.0, 1.0)
    w_h, w_c, w_t, w_v = 0.40, 0.30, 0.20, 0.10
    penalty = (w_h*h_n + w_c*c_n + w_t*t_n + w_v*v_n) * 100.0
    score = max(0.0, 100.0 - penalty)
    due = (score < 60.0) or (max_temp_c > 75.0) or (vib_g_rms > 1.2)
    return score, penalty, t_excess, due

def mission_rating(go, E_need, E_avail, health_score):
    if not go: return "F ❌"
    eff = E_need / max(1e-6, E_avail)
    if eff < 0.5 and health_score > 80: return "A ✅"
    if eff < 0.7 and health_score > 70: return "B 👍"
    if eff < 0.9 and health_score > 60: return "C ⚠️"
    return "D ❗"

# ── Higher-fidelity energy model (keeps original intact) ─────────────────────
def energy_wh_better(
    distance_m: float,
    V_ms: float,
    mass_kg: float,
    hover_to_s: float,
    hover_ldg_s: float,
    rho: float = 1.225,
    S_ref: float = 3.5,
    CD0: float = 0.08,
    AR: float = 8.0,
    e: float = 0.8,
    A_disk_total: float = 20.0,
    eta_prop: float = 0.75,
    eta_elec: float = 0.93,
    loiter_min: float = 0.0,
    loiter_mode: str = "hover"  # "hover" or "cruise"
):
    g = 9.80665
    W = mass_kg * g
    # Hover power (air) via momentum theory
    P_hover_air = (W ** 1.5) / max(1e-6, (2.0 * rho * A_disk_total) ** 0.5)
    P_hover_elec = P_hover_air / max(1e-6, (eta_prop * eta_elec))
    E_hover_Wh = P_hover_elec * (hover_to_s + hover_ldg_s) / 3600.0
    # Cruise power (air): parasite + induced
    P_par_air = 0.5 * rho * (V_ms ** 3) * S_ref * CD0
    k = 1.0 / max(1e-6, (math.pi * e * AR))
    P_ind_air = (2.0 * k * (W ** 2)) / max(1e-6, (rho * V_ms * S_ref))
    P_cruise_air = P_par_air + P_ind_air
    P_cruise_elec = P_cruise_air / max(1e-6, (eta_prop * eta_elec))
    # Cruise time & energy
    t_cruise_s = distance_m / max(1e-6, V_ms)
    E_cruise_Wh = P_cruise_elec * t_cruise_s / 3600.0
    # Loiter reserve
    if loiter_mode == "cruise":
        E_loiter_Wh = P_cruise_elec * (loiter_min * 60.0) / 3600.0
    else:
        E_loiter_Wh = P_hover_elec * (loiter_min * 60.0) / 3600.0
    E_total_Wh = E_hover_Wh + E_cruise_Wh + E_loiter_Wh
    return E_total_Wh, P_hover_elec, P_cruise_elec, t_cruise_s, E_hover_Wh, E_cruise_Wh, E_loiter_Wh

# ─────────────────────────────────────────────────────────
# Lightweight event logger (exportable)
# ─────────────────────────────────────────────────────────
st.session_state.setdefault("logs", [])

def log_event(event: str, **fields):
    st.session_state["logs"].append({
        "ts": datetime.utcnow().isoformat() + "Z",
        "event": event,
        **fields
    })

def logs_dataframe() -> pd.DataFrame:
    if not st.session_state["logs"]:
        return pd.DataFrame(columns=["ts", "event"])
    return pd.DataFrame(st.session_state["logs"])

# ─────────────────────────────────────────────────────────
# Session + Sidebar HUD
# ─────────────────────────────────────────────────────────
st.session_state.setdefault("health_now", 80.0)
st.session_state.setdefault("energy_now", None)

st.sidebar.title("HUD")
sb_health = st.sidebar.empty()
sb_energy = st.sidebar.empty()
sb_status = st.sidebar.empty()
sb_health.metric("Health", f"{st.session_state['health_now']:.1f} / 100")
sb_energy.write("Remaining Energy: —")
sb_status.write("Status: Ready")

# Logging toggles
st.sidebar.subheader("Logging")
st.sidebar.caption("Automatically record planner runs & ratings")
st.session_state.setdefault("log_auto_flight", True)
st.session_state.setdefault("log_auto_rating", True)
st.session_state["log_auto_flight"] = st.sidebar.toggle(
    "Auto-log flight runs", value=st.session_state["log_auto_flight"], key="log_auto_flight_toggle"
)
st.session_state["log_auto_rating"] = st.sidebar.toggle(
    "Auto-log mission rating", value=st.session_state["log_auto_rating"], key="log_auto_rating_toggle"
)

# Mobile-friendly packing for controls
compact = st.sidebar.toggle("Compact controls (mobile)", value=True, key="ui_compact")

# ─────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "City Hop Planner", "Perception Sandbox", "Health Monitor",
    "Fleet UTM", "Copilot"
])

# ============ 1) City Hop Planner ============
with tab1:
    st.subheader("City Hop Planner (with playback ✈️)")

    if compact:
        grid_W = st.slider("Grid width",20,60,40,key="plan_w")
        grid_H = st.slider("Grid height",20,60,40,key="plan_h")
        scale_m = st.selectbox("Meters/cell",[25,50,75,100],index=1,key="plan_scale")
        reserve_pct = st.slider("Reserve %",0,30,15,key="plan_reserve")
    else:
        c = st.columns(4)
        grid_W=c[0].slider("Grid width",20,60,40,key="plan_w")
        grid_H=c[1].slider("Grid height",20,60,40,key="plan_h")
        scale_m=c[2].selectbox("Meters/cell",[25,50,75,100],index=1,key="plan_scale")
        reserve_pct=c[3].slider("Reserve %",0,30,15,key="plan_reserve")

    if compact:
        mass = st.number_input("Mass (kg)",300.0,5000.0,1100.0,step=50.0,key="plan_mass")
        cruise = st.number_input("Cruise speed (m/s)",20.0,120.0,45.0,step=5.0,key="plan_cruise")
        hover_s = st.number_input("Hover time (s)",0.0,300.0,60.0,step=5.0,key="plan_hover")
        batt = st.number_input("Battery (Wh)",50000.0,400000.0,120000.0,step=5000.0,key="plan_batt")
    else:
        c = st.columns(4)
        mass=c[0].number_input("Mass (kg)",300.0,5000.0,1100.0,step=50.0,key="plan_mass")
        cruise=c[1].number_input("Cruise speed (m/s)",20.0,120.0,45.0,step=5.0,key="plan_cruise")
        hover_s=c[2].number_input("Hover time (s)",0.0,300.0,60.0,step=5.0,key="plan_hover")
        batt=c[3].number_input("Battery (Wh)",50000.0,400000.0,120000.0,step=5000.0,key="plan_batt")

    c = st.columns(4)
    start=(int(c[0].number_input("Start Y",0,grid_H-1,5,key="plan_sy")),
           int(c[1].number_input("Start X",0,grid_W-1,5,key="plan_sx")))
    goal=(int(c[2].number_input("Goal Y",0,grid_H-1,grid_H-6,key="plan_gy")),
          int(c[3].number_input("Goal X",0,grid_W-1,grid_W-6,key="plan_gx")))

    nfz_rects=[]
    nfz_count=st.slider("No-Fly Zones",0,3,1,key="plan_nfz_n")
    for i in range(nfz_count):
        with st.expander(f"NFZ #{i+1}",expanded=True):
            c=st.columns(4)
            y0=c[0].number_input("y0",0,grid_H-1,10,key=f"plan_nfz_y0_{i}")
            x0=c[1].number_input("x0",0,grid_W-1,10,key=f"plan_nfz_x0_{i}")
            y1=c[2].number_input("y1",0,grid_H-1,15,key=f"plan_nfz_y1_{i}")
            x1=c[3].number_input("x1",0,grid_W-1,15,key=f"plan_nfz_x1_{i}")
            y0,y1=sorted([y0,y1]); x0,x1=sorted([x0,x1])
            nfz_rects.append((y0,x0,y1,x1))

    grid=np.zeros((grid_H,grid_W),dtype=int)
    for (y0,x0,y1,x1) in nfz_rects: grid[y0:y1+1,x0:x1+1]=1

    path=astar(grid,start,goal)
    distance=((len(path)-1) if path else 0)*scale_m

    # Physics fidelity
    fidelity = st.selectbox(
        "Physics fidelity",
        ["Simple (demo)", "Better (induced + parasite, efficiencies)"],
        index=0,
        key="plan_fidelity"
    )

    if fidelity.startswith("Better"):
        with st.expander("Better physics settings", expanded=False):
            cA = st.columns(4)
            hover_to_s   = cA[0].number_input("Takeoff hover (s)", 0.0, 600.0, 45.0, 5.0, key="plan_to_hover")
            hover_ldg_s  = cA[1].number_input("Landing hover (s)", 0.0, 600.0, 45.0, 5.0, key="plan_ldg_hover")
            loiter_min   = cA[2].number_input("Loiter reserve (min)", 0.0, 30.0, 3.0, 1.0, key="plan_loiter_min")
            loiter_mode  = cA[3].selectbox("Loiter mode", ["hover", "cruise"], index=0, key="plan_loiter_mode")
            cB = st.columns(4)
            rho        = cB[0].number_input("Air density ρ (kg/m³)", 0.8, 1.6, 1.225, 0.005, key="plan_rho")
            S_ref      = cB[1].number_input("Ref area S (m²)", 0.5, 20.0, 3.5, 0.1, key="plan_S")
            CD0        = cB[2].number_input("CD₀ (parasite)", 0.02, 0.30, 0.08, 0.005, key="plan_CD0")
            A_disk_tot = cB[3].number_input("Rotor disk area ΣA (m²)", 2.0, 150.0, 20.0, 0.5, key="plan_Adisk")
            cC = st.columns(4)
            AR    = cC[0].number_input("Aspect ratio AR", 3.0, 20.0, 8.0, 0.5, key="plan_AR")
            e     = cC[1].number_input("Oswald e", 0.5, 1.0, 0.80, 0.01, key="plan_e")
            eta_p = cC[2].number_input("Prop eff ηₚ", 0.5, 0.95, 0.75, 0.01, key="plan_eta_p")
            eta_e = cC[3].number_input("Elec eff ηₑ", 0.5, 0.99, 0.93, 0.01, key="plan_eta_e")

    # Energy & GO (fidelity-aware)
    if fidelity.startswith("Better"):
        E_need, P_h, P_c, t_c, E_hover_Wh, E_cruise_Wh, E_loiter_Wh = energy_wh_better(
            distance_m=distance,
            V_ms=cruise,
            mass_kg=mass,
            hover_to_s=hover_to_s,
            hover_ldg_s=hover_ldg_s,
            rho=rho, S_ref=S_ref, CD0=CD0, AR=AR, e=e,
            A_disk_total=A_disk_tot,
            eta_prop=eta_p, eta_elec=eta_e,
            loiter_min=loiter_min, loiter_mode=loiter_mode
        )
        reserve=(reserve_pct/100.0)*batt
    else:
        E_need,P_h,P_c,t_c=energy_wh(distance,cruise,mass,hover_s)
        reserve=(reserve_pct/100.0)*batt

    go=(path is not None) and (E_need + reserve <= batt)

    # Auto snapshot of planning context
    if st.session_state.get("log_auto_flight", True):
        log_event(
            "plan_snapshot",
            grid_W=int(grid_W), grid_H=int(grid_H), scale_m=int(scale_m),
            start=start, goal=goal, nfz_count=int(nfz_count),
            distance_m=float(distance), energy_need_Wh=float(E_need),
            battery_Wh=float(batt), reserve_Wh=float(reserve), go=bool(go),
            mass_kg=float(mass), cruise_ms=float(cruise), hover_s=float(hover_s),
            fidelity=fidelity
        )

    # Plot
    fig, ax = draw_grid(grid,start,goal,nfz_rects,scale_m, title="City Grid")
    if path:
        xs=[x for y,x in path]; ys=[y for y,x in path]
        ax.plot(xs,ys,linewidth=2)
    st.pyplot(fig)

    # Metrics
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Distance",f"{distance/1000:.2f} km")
    c2.metric("Hover power",f"{P_h/1000:.1f} kW")
    c3.metric("Cruise power",f"{P_c/1000:.1f} kW")
    c4.metric("Cruise time",f"{t_c/60:.1f} min")
    if fidelity.startswith("Better"):
        st.caption(f"E_hover={E_hover_Wh:,.0f} Wh | E_cruise={E_cruise_Wh:,.0f} Wh | E_loiter={E_loiter_Wh:,.0f} Wh")
    st.info(f"Need: {E_need:,.0f} Wh | Battery: {batt:,.0f} Wh | Reserve: {reserve:,.0f} Wh")
    st.success("GO ✅") if go else st.error("NO-GO ❌")

    # Manual snapshot
    if st.button("📝 Log plan snapshot", key="plan_log"):
        log_event(
            "plan_snapshot_manual",
            grid_W=int(grid_W), grid_H=int(grid_H), scale_m=int(scale_m),
            start=start, goal=goal, nfz_count=int(nfz_count),
            distance_m=float(distance), energy_need_Wh=float(E_need),
            battery_Wh=float(batt), reserve_Wh=float(reserve), go=bool(go),
            mass_kg=float(mass), cruise_ms=float(cruise), hover_s=float(hover_s),
            fidelity=fidelity
        )
        st.success("Logged plan snapshot.")

    sb_health.metric("Health", f"{st.session_state.get('health_now',80.0):.1f} / 100")
    sb_status.write("Status: Planned ✅" if go else "Status: Blocked / Low energy ❌")

    # Playback
    if go and st.button("▶️ Play flight", key="plan_play"):
        placeholder_metric = st.empty()
        placeholder_plot = st.empty()
        energy=batt
        steps=max(1,len(path))
        st.session_state["energy_now"] = energy
        sb_energy.metric("Remaining Energy", f"{energy:,.0f} Wh")
        sb_status.write("Status: In flight ✈️")

        # Auto-log: flight start
        if st.session_state.get("log_auto_flight", True):
            log_event(
                "flight_started",
                steps=int(steps),
                energy_start_Wh=float(batt),
                grid_W=int(grid_W), grid_H=int(grid_H), scale_m=int(scale_m),
                start=start, goal=goal, nfz_count=int(nfz_count),
                mass_kg=float(mass), cruise_ms=float(cruise), hover_s=float(hover_s),
                reserve_pct=float(reserve_pct),
                fidelity=fidelity
            )

        for i,(y,x) in enumerate(path):
            fig, ax = draw_grid(grid,start,goal,nfz_rects,scale_m, title="Flight Playback")
            xs=[p[1] for p in path[:i+1]]; ys=[p[0] for p in path[:i+1]]
            ax.plot(xs,ys,linewidth=2); ax.scatter(x,y,c="red",s=60)
            placeholder_plot.pyplot(fig)

            energy -= E_need/steps
            st.session_state["energy_now"] = max(0.0, energy)
            placeholder_metric.metric("Remaining energy", f"{energy:,.0f} Wh")
            sb_energy.metric("Remaining Energy", f"{energy:,.0f} Wh")
            time.sleep(0.08)

        # Auto-log: flight end
        if st.session_state.get("log_auto_flight", True):
            log_event(
                "flight_completed",
                energy_end_Wh=float(st.session_state.get("energy_now", 0.0)),
                distance_m=float(((len(path)-1) if path else 0) * scale_m),
                energy_need_Wh=float(E_need),
                battery_Wh=float(batt),
                reserve_Wh=float(reserve),
                fidelity=fidelity
            )

        health_now = st.session_state.get("health_now", 80.0)
        rating = mission_rating(go, E_need, batt, health_now)
        st.subheader(f"Mission Rating: {rating}")
        st.caption(f"(Uses current Health score: {health_now:.1f}/100)")
        sb_status.write(f"Status: Completed — Rating {rating}")

        # Auto-log: rating
        if st.session_state.get("log_auto_rating", True):
            log_event("mission_rating", rating=str(rating), health=float(health_now), go=bool(go), fidelity=fidelity)

# ============ 2) Perception Sandbox ============
with tab2:
    st.subheader("Perception Sandbox (toy lidar)")

    if compact:
        num_obs = st.slider("Obstacles", 2, 25, 8, key="perc_obs")
        field = st.selectbox("Field size (m)", [30, 40, 60], index=1, key="perc_field")
        r_max = st.slider("Lidar range (m)", 5, 40, 20, key="perc_rmax")
        safety_radius = st.slider("Safety stop radius (m)", 1, 10, 5, key="perc_safe")
        rays = st.slider("Rays (angles)", 60, 360, 180, step=10, key="perc_rays")
        res = st.slider("Ray step resolution (m)", 0.1, 1.0, 0.25, step=0.05, key="perc_res")
        seed = st.slider("Random seed", 0, 999, 7, key="perc_seed")
    else:
        c = st.columns(3)
        num_obs = c[0].slider("Obstacles", 2, 25, 8, key="perc_obs")
        field   = c[1].selectbox("Field size (m)", [30, 40, 60], index=1, key="perc_field")
        r_max   = c[2].slider("Lidar range (m)", 5, 40, 20, key="perc_rmax")
        c = st.columns(4)
        safety_radius = c[0].slider("Safety stop radius (m)", 1, 10, 5, key="perc_safe")
        rays          = c[1].slider("Rays (angles)", 60, 360, 180, step=10, key="perc_rays")
        res           = c[2].slider("Ray step resolution (m)", 0.1, 1.0, 0.25, step=0.05, key="perc_res")
        seed          = c[3].slider("Random seed", 0, 999, 7, key="perc_seed")

    rng = np.random.default_rng(seed)
    obs = rng.uniform(low=-field/2, high=field/2, size=(num_obs, 2))
    vx, vy = 0.0, 0.0

    angles = np.linspace(0, 2*np.pi, rays, endpoint=False)
    hits = []
    for th in angles:
        d = 0.0; hit = r_max
        while d <= r_max:
            x = vx + d*math.cos(th); y = vy + d*math.sin(th)
            if np.any(np.hypot(obs[:,0]-x, obs[:,1]-y) < 0.6):
                hit = d; break
            d += res
        hits.append(hit)
    hits = np.array(hits)
    nearest = hits.min() if len(hits) else r_max
    brake = nearest < safety_radius

    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(obs[:,0], obs[:,1], marker="s")
    ax.scatter([vx], [vy], c="red")
    for d,th in zip(hits, angles):
        ax.plot([vx, vx + d*math.cos(th)], [vy, vy + d*math.sin(th)], linewidth=0.5)
    safe = plt.Circle((vx,vy), safety_radius, fill=False, linestyle="--")
    ax.add_patch(safe)
    ax.set_aspect("equal", "box"); ax.grid(True, alpha=0.2)
    st.pyplot(fig)

    c1,c2,c3 = st.columns(3)
    c1.metric("Nearest obstacle", f"{nearest:.2f} m")
    c2.metric("Rays with hit", f"{int(np.sum(hits < r_max))}")
    c3.metric("Brake/avoid", "YES" if brake else "NO")

    # Manual log button
    if st.button("📝 Log perception snapshot", key="perc_log"):
        log_event(
            "perception_snapshot",
            obstacles=int(num_obs), field_m=int(field), lidar_range_m=int(r_max),
            safety_radius_m=int(safety_radius), rays=int(rays), resolution_m=float(res),
            seed=int(seed), nearest_m=float(nearest),
            rays_hit=int(int(np.sum(hits < r_max))), brake=bool(brake)
        )
        st.success("Logged perception snapshot.")

# ============ 3) Health Monitor ============
with tab3:
    st.subheader("Predictive Maintenance (toy)")

    # sensible defaults the first time
    st.session_state.setdefault("hm_hours", 300.0)
    st.session_state.setdefault("hm_cycles", 250.0)
    st.session_state.setdefault("hm_temp", 50.0)
    st.session_state.setdefault("hm_vib", 0.3)

    bcol = st.columns(3)
    if bcol[0].button("Reset to typical", key="hm_reset"):
        st.session_state["hm_hours"] = 300.0
        st.session_state["hm_cycles"] = 250.0
        st.session_state["hm_temp"]  = 50.0
        st.session_state["hm_vib"]   = 0.3

    if bcol[1].button("Maintenance: Battery swap", key="hm_bswap"):
        st.session_state["hm_cycles"] = 0.0

    if bcol[2].button("Maintenance: Service pack", key="hm_service"):
        st.session_state["hm_hours"] = max(0.0, st.session_state["hm_hours"] - 150.0)
        st.session_state["hm_temp"]  = max(40.0, st.session_state["hm_temp"]  - 8.0)
        st.session_state["hm_vib"]   = max(0.0, st.session_state["hm_vib"]   - 0.2)

    c = st.columns(4)
    hours = c[0].number_input("Flight hours", 0.0, 5000.0, st.session_state["hm_hours"], step=10.0, key="hm_hours")
    cycles = c[1].number_input("Charge cycles", 0.0, 6000.0, st.session_state["hm_cycles"], step=10.0, key="hm_cycles")
    temp   = c[2].number_input("Max pack temp (°C)", 0.0, 120.0, st.session_state["hm_temp"], step=1.0, key="hm_temp")
    vib    = c[3].number_input("Vibration gRMS", 0.0, 5.0, st.session_state["hm_vib"], step=0.1, key="hm_vib")

    score, penalty, excess_temp, due = compute_health(hours, cycles, temp, vib)

    st.session_state["health_now"] = float(score)  # share to HUD / Planner
    sb_health.metric("Health", f"{st.session_state['health_now']:.1f} / 100")

    st.metric("Health score (now)", f"{score:,.1f} / 100")
    st.progress(int(score))
    st.caption(
        f"Penalty → hours: {min(hours/2000,1)*40:.1f}, "
        f"cycles: {min(cycles/2000,1)*30:.1f}, "
        f"temp>40°C: {min(excess_temp/40,1)*20:.1f}, "
        f"vibration: {min(vib/2,1)*10:.1f}  |  Total: {penalty:.1f}"
    )
    st.write("• Maintenance window:", "**Soon**" if due else "**Not yet due**")

    # Manual logs
    if st.button("📝 Log current health", key="hm_log_now"):
        log_event(
            "health_now",
            hours=float(hours), cycles=float(cycles),
            max_temp_c=float(temp), vib_g_rms=float(vib),
            score=float(score), due=bool(due), penalty=float(penalty)
        )
        st.success("Logged current health.")

    st.divider()
    st.subheader("What-if: before next check")

    w = st.columns(4)
    add_hours  = w[0].slider("Add hours", 0.0, 300.0, 25.0, step=1.0, key="hm_add_hours")
    add_cycles = w[1].slider("Add cycles", 0.0, 1000.0, 60.0, step=5.0, key="hm_add_cycles")
    temp_bias  = w[2].slider("Temp increase (°C)", -10.0, 30.0, 5.0, step=1.0, key="hm_temp_bias")
    vib_bias   = w[3].slider("Vibration change (gRMS)", -0.5, 1.5, 0.2, step=0.1, key="hm_vib_bias")

    fut_score, fut_penalty, fut_excess, fut_due = compute_health(
        hours + add_hours,
        cycles + add_cycles,
        temp + temp_bias,
        max(0.0, vib + vib_bias)
    )

    colA, colB = st.columns(2)
    with colA:
        st.metric("Projected score", f"{fut_score:,.1f} / 100", delta=f"{fut_score - score:+.1f}")
        st.progress(int(fut_score))
    with colB:
        st.write("• Projected maintenance window:", "**Soon**" if fut_due else "**Not yet due**")
        st.caption(
            f"Projected → hours: {min((hours+add_hours)/2000,1)*40:.1f}, "
            f"cycles: {min((cycles+add_cycles)/2000,1)*30:.1f}, "
            f"temp>40°C: {min(fut_excess/40,1)*20:.1f}, "
            f"vibration: {min(max(0.0, vib+vib_bias)/2,1)*10:.1f}  |  Total: {fut_penalty:.1f}"
        )

    if st.button("📝 Log projection", key="hm_log_proj"):
        log_event(
            "health_projection",
            add_hours=float(add_hours), add_cycles=float(add_cycles),
            temp_bias=float(temp_bias), vib_bias=float(vib_bias),
            projected_score=float(fut_score), projected_due=bool(fut_due),
            projected_penalty=float(fut_penalty)
        )
        st.success("Logged projected health.")

# ============ 4) Fleet UTM (multi-vehicle coordination) ============
with tab4:
    st.subheader("Fleet UTM (multi-vehicle, adaptive coordination)")

    # Grid + NFZ
    fW = st.slider("Grid width", 20, 60, 40, key="utm_w")
    fH = st.slider("Grid height", 20, 60, 40, key="utm_h")
    fScale = st.selectbox("Meters/cell", [25, 50, 75, 100], index=1, key="utm_scale")

    nfz_rects_f = []
    nfz_n = st.slider("No-Fly Zones", 0, 3, 1, key="utm_nfz_n")
    for i in range(nfz_n):
        with st.expander(f"NFZ #{i+1}", expanded=False):
            c = st.columns(4)
            y0 = c[0].number_input("y0", 0, fH-1, 10, key=f"utm_nfz_y0_{i}")
            x0 = c[1].number_input("x0", 0, fW-1, 10, key=f"utm_nfz_x0_{i}")
            y1 = c[2].number_input("y1", 0, fH-1, 15, key=f"utm_nfz_y1_{i}")
            x1 = c[3].number_input("x1", 0, fW-1, 15, key=f"utm_nfz_x1_{i}")
            y0, y1 = sorted([y0, y1]); x0, x1 = sorted([x0, x1])
            nfz_rects_f.append((y0, x0, y1, x1))

    gridF = np.zeros((fH, fW), dtype=int)
    for (y0,x0,y1,x1) in nfz_rects_f: gridF[y0:y1+1, x0:x1+1] = 1

    # Fleet + policy
    cols = st.columns(4)
    N = cols[0].slider("Vehicles", 2, 10, 5, key="utm_N")
    base_speed = cols[1].selectbox("Base speed (cells/step)", [1, 2], index=0, key="utm_speed")
    adaptive = cols[2].toggle("Adaptive coordination", value=True, key="utm_adapt")
    steps_max = cols[3].slider("Max steps", 50, 500, 180, key="utm_steps")

    st.markdown("**Sensor Fusion Confidence (toy: lidar/radar/camera fusion)**")
    c = st.columns(3)
    conf_lidar = c[0].slider("Lidar conf", 0.0, 1.0, 0.9, 0.01, key="utm_conf_l")
    conf_radar = c[1].slider("Radar conf", 0.0, 1.0, 0.8, 0.01, key="utm_conf_r")
    conf_cam   = c[2].slider("Camera conf", 0.0, 1.0, 0.7, 0.01, key="utm_conf_c")
    fusion_conf = float(np.clip(0.5*conf_lidar + 0.3*conf_radar + 0.2*conf_cam, 0.0, 1.0))

    health_now = float(st.session_state.get("health_now", 80.0)) / 100.0

    seed = st.slider("Random seed", 0, 999, 11, key="utm_seed")
    rng = np.random.default_rng(seed)
    free = np.argwhere(gridF == 0)
    if len(free) < 2*N:
        st.error("Not enough free cells for starts/goals — reduce NFZs or increase grid size.")
        st.stop()
    rng.shuffle(free)
    starts = [tuple(map(int, free[i])) for i in range(N)]
    goals  = [tuple(map(int, free[-(i+1)])) for i in range(N)]

    sep_base = 1
    sep = sep_base + (0 if not adaptive else int(np.interp(1 - min(fusion_conf, health_now), [0,1], [0,2])))
    lam_conflict = 10.0 if not adaptive else float(np.interp(1 - fusion_conf, [0,1], [4.0, 18.0]))
    lam_delay    = 0.1 if not adaptive else float(np.interp(health_now, [0,1], [0.25, 0.05]))
    lam_nfz      = 50.0

    def cell_free(y,x): return 0 <= y < fH and 0 <= x < fW and gridF[y,x]==0
    def neighbors(y,x):
        for dy,dx in [(1,0),(-1,0),(0,1),(0,-1)]:
            ny,nx = y+dy, x+dx
            if cell_free(ny,nx): yield (ny,nx)

    def next_step(pos, others, goal):
        def manh(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
        best, best_cost = pos, 1e9
        for ny, nx in list(neighbors(*pos)) + [pos]:
            conflict = sum(1 for o in others if (abs(o[0]-ny)+abs(o[1]-nx)) <= sep and o != pos)
            nfz_cross = 0 if gridF[ny,nx]==0 else 1
            cost = manh((ny,nx), goal) + lam_conflict*conflict + lam_nfz*nfz_cross + lam_delay
            if cost < best_cost: best, best_cost = (ny,nx), cost
        return best

    run = st.button("Run simulation", key="utm_run")
    plot_ph = st.empty(); metrics_ph = st.empty()
    positions = starts[:]; reached = np.zeros(N, dtype=bool); los_viol = 0

    if run:
        log_event("fleet_sim_started", N=int(N), sep=int(sep), adaptive=bool(adaptive),
                  fusion_conf=float(fusion_conf), health=float(health_now))

        for t in range(steps_max):
            new_pos=[]
            for i,p in enumerate(positions):
                if reached[i]: new_pos.append(p); continue
                others = positions[:i] + positions[i+1:]
                step = p
                for _ in range(base_speed):  # multi-cells per step
                    step = next_step(step, others, goals[i])
                new_pos.append(step)
                if step == goals[i]: reached[i]=True
            positions = new_pos

            for i in range(N):
                for j in range(i+1, N):
                    if (abs(positions[i][0]-positions[j][0]) + abs(positions[i][1]-positions[j][1])) < sep:
                        los_viol += 1

            fig, ax = draw_grid(gridF, (-1,-1), (-1,-1), nfz_rects_f, fScale, title="Fleet UTM")
            xs=[x for y,x in positions]; ys=[y for y,x in positions]
            gx=[x for y,x in goals]; gy=[y for y,x in goals]
            ax.scatter(xs, ys, s=60); ax.scatter(gx, gy, marker="X", s=70)
            plot_ph.pyplot(fig)

            eta_cells = np.mean([abs(p[0]-g[0]) + abs(p[1]-g[1]) for p,g in zip(positions, goals)])
            done = int(np.sum(reached))
            metrics_ph.write(
                f"Step {t+1}/{steps_max} | Reached {done}/{N} | Mean remaining {eta_cells:.1f} cells | "
                f"Sep buffer {sep} | LOS violations {los_viol}"
            )
            if done == N: break

        log_event("fleet_sim_completed", N=int(N), sep=int(sep), adaptive=bool(adaptive),
                  fusion_conf=float(fusion_conf), health=float(health_now),
                  los_violations=int(los_viol), reached=int(np.sum(reached)))
        st.success("Simulation finished ✅")
        st.caption("Tip: open **📜 Logs & Export** below to download results.")

# ============ 5) Copilot (heuristic by default; optional cloud LLM) ============
with tab5:
    st.subheader("Copilot (log-aware helper)")
    st.caption("Default: local heuristic (no network). Optionally enable a cloud LLM with an API key.")

    # Controls
    enable_cloud_llm = st.toggle("Enable cloud LLM (optional)", value=False, help="Off = local heuristic only")
    api_key_input = ""
    if enable_cloud_llm:
        api_key_input = st.text_input("OpenAI API key", type="password", help="Or set OPENAI_API_KEY env var")
    api_key_env = os.getenv("OPENAI_API_KEY", "") if enable_cloud_llm else ""
    api_key = (api_key_input.strip() or api_key_env) if enable_cloud_llm else ""

    # Chat history
    st.session_state.setdefault("chat_msgs", [])
    for role, content in st.session_state["chat_msgs"]:
        with st.chat_message(role):
            st.write(content)

    user_msg = st.chat_input("Ask: 'Why NO-GO?', 'Summarize', 'Fleet results?'")
    if user_msg:
        st.session_state["chat_msgs"].append(("user", user_msg))
        with st.chat_message("user"): st.write(user_msg)

        # Context from latest logs
        df_logs = logs_dataframe()
        last_plan  = df_logs[df_logs["event"].str.contains("plan_snapshot")].tail(1).to_dict("records")
        last_rate  = df_logs[df_logs["event"]=="mission_rating"].tail(1).to_dict("records")
        last_fleet = df_logs[df_logs["event"]=="fleet_sim_completed"].tail(1).to_dict("records")
        context = {
            "last_plan":  last_plan[0] if last_plan else {},
            "last_rating":last_rate[0] if last_rate else {},
            "last_fleet": last_fleet[0] if last_fleet else {},
        }

        reply = None
        mode = "heuristic"

        # Optional cloud LLM
        if enable_cloud_llm and api_key:
            try:
                import requests
                prompt = (
                    "You are a concise aerospace copilot. Use the JSON context to answer.\n"
                    f"User: {user_msg}\n\n"
                    f"Context JSON:\n{json.dumps(context, indent=2)}\n\n"
                    "Answer in 1-4 short bullets. If data is missing, say what to run in the app to produce it."
                )
                res = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"},
                    json={"model": "gpt-4o-mini", "messages":[{"role":"user","content":prompt}], "temperature":0.2},
                    timeout=20
                )
                data = res.json()
                reply = data["choices"][0]["message"]["content"].strip()
                mode = "openai"
            except Exception:
                reply = None
                mode = "heuristic"

        # Heuristic fallback / default
        if not reply:
            lines = []
            q = user_msg.lower()
            p = context["last_plan"]; r = context["last_rating"]; f = context["last_fleet"]

            def have_plan(): return bool(p)
            def have_fleet(): return bool(f)
            def have_rating(): return bool(r)

            if "summary" in q or "summarize" in q:
                if have_plan():
                    try:
                        lines.append(
                            f"• Plan: {p.get('distance_m',0):.0f} m, Need {p.get('energy_need_Wh',0):.0f} Wh, "
                            f"Reserve {p.get('reserve_Wh',0):.0f} Wh → GO={p.get('go')}"
                        )
                    except Exception:
                        lines.append("• Plan: summary available (some fields missing).")
                if have_rating():
                    lines.append(f"• Mission rating: {r.get('rating')} (Health {r.get('health')})")
                if have_fleet():
                    lines.append(f"• Fleet: N={f.get('N')}, LOS={f.get('los_violations')}, Reached={f.get('reached')}")
                if not lines:
                    lines.append("• No recent logs. Run a plan/flight or fleet sim, then ask again.")
            elif "why no-go" in q or "no-go" in q:
                if have_plan():
                    need, reserve, batt = p.get("energy_need_Wh"), p.get("reserve_Wh"), p.get("battery_Wh")
                    if None not in (need, reserve, batt) and (need + reserve) > batt:
                        lines.append("• Need + Reserve > Battery. Reduce reserve %, distance, mass, hover, or increase battery.")
                    else:
                        lines.append("• Last snapshot doesn't show NO-GO. Re-run planner and snapshot again.")
                else:
                    lines.append("• No plan snapshot found. Run a plan and click Log snapshot.")
            elif "fleet" in q or "los" in q or "coordination" in q:
                if have_fleet():
                    lines.append(f"• Last fleet run: N={f.get('N')}, LOS={f.get('los_violations')}, "
                                 f"Adaptive={f.get('adaptive')}, Sep={f.get('sep')}")
                    lines.append("• Tip: raise fusion confidence or separation; enable Adaptive to reduce LOS.")
                else:
                    lines.append("• No fleet run logged yet. Open Fleet UTM → Run simulation → ask again.")
            else:
                lines.append("• Try: 'Summarize', 'Why NO-GO?', or 'Fleet results?'.")
                lines.append("• You can also ask about health, reserves, or energy drivers.")
            reply = "\n".join(lines)

        with st.chat_message("assistant"):
            st.write(reply)
        st.session_state["chat_msgs"].append(("assistant", reply))
        log_event("copilot_exchange", mode=mode, question=user_msg, answer=reply)

# ─────────────────────────────────────────────────────────
# 📜 Logs & Export
# ─────────────────────────────────────────────────────────
with st.expander("📜 Logs & Export", expanded=False):
    df = logs_dataframe()
    st.caption(f"{len(df)} events recorded")
    if len(df):
        st.dataframe(df, use_container_width=True, height=260)
        csv_str = df.to_csv(index=False)
        json_str = json.dumps(st.session_state["logs"], indent=2)
        st.download_button(
            "⬇️ Download CSV",
            data=csv_str.encode("utf-8"),
            file_name=f"evtol_logs_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv",
            mime="text/csv",
            key="logs_dl_csv",
        )
        st.download_button(
            "⬇️ Download JSON",
            data=json_str.encode("utf-8"),
            file_name=f"evtol_logs_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json",
            mime="application/json",
            key="logs_dl_json",
        )
        st.markdown("**Copy to clipboard → JSON**")
        st.code(json_str, language="json")
        st.markdown("**Copy to clipboard → CSV**")
        st.code(csv_str, language="csv")
        if st.button("🗑️ Clear all logs", key="logs_clear"):
            st.session_state["logs"] = []
            st.rerun()
    else:
        st.info("No events yet. Use the snapshot buttons above, or run a flight / fleet sim.")
