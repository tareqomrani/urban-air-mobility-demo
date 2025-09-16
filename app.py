# eVTOL Mini-Lab — ultra-simple demos for autonomy, perception, and maintenance
# Run: streamlit run app.py

import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from typing import Tuple, List

st.set_page_config(page_title="eVTOL Mini-Lab", page_icon="🛩️", layout="wide")
st.title("🛩️ eVTOL Mini-Lab")
st.caption("Tiny, friendly demos: route planning, obstacle perception, and maintenance health.")

# ─────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# A* on a small grid
def astar(grid: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]):
    H, W = grid.shape
    def h(a,b):  # manhattan
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
    open_set = {start}
    came_from = {}
    g = {start: 0}
    f = {start: h(start, goal)}
    while open_set:
        current = min(open_set, key=lambda n: f.get(n, 1e9))
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
    # explore neighbors
        open_set.remove(current)
        for dy,dx in [(1,0),(-1,0),(0,1),(0,-1)]:
            ny, nx = current[0]+dy, current[1]+dx
            if not (0 <= ny < H and 0 <= nx < W): 
                continue
            if grid[ny, nx] == 1:  # blocked
                continue
            tentative = g[current] + 1
            if tentative < g.get((ny,nx), 1e9):
                came_from[(ny,nx)] = current
                g[(ny,nx)] = tentative
                f[(ny,nx)] = tentative + h((ny,nx), goal)
                open_set.add((ny,nx))
    return None

def draw_grid_path(grid, path, start, goal, nfz_rects, scale_m=50, ax=None, title=""):
    H, W = grid.shape
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(grid, cmap="Greys", origin="lower")
    for (y,x) in [start, goal]:
        ax.scatter(x, y, s=60, marker="o" if (y,x)==start else "X")
    for (y0,x0,y1,x1) in nfz_rects:
        rect = plt.Rectangle((x0, y0), x1-x0+1, y1-y0+1, fill=True, alpha=0.2)
        ax.add_patch(rect)
    if path:
        xs = [x for y,x in path]
        ys = [y for y,x in path]
        ax.plot(xs, ys, linewidth=3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title + f" (scale ~{scale_m} m/cell)")
    return ax

# Simple hover+cruise energy model (toy)
def energy_wh(distance_m, cruise_ms, mass_kg, hover_s, disk_loading_proxy=30.0, CdA=0.6, rho=1.2, eff=0.75):
    # Hover power ~ k1 * mass^(3/2) (toy) with loading factor
    k_hover = 25.0
    P_hover = k_hover * (mass_kg ** 1.5) * (1.0 + 0.015*disk_loading_proxy)  # W
    E_hover = P_hover * hover_s / 3600.0
    # Parasitic power ~ 0.5 rho V^3 CdA / eff (toy)
    P_cruise = (0.5 * rho * (cruise_ms**3) * CdA) / max(0.3, eff)
    t_cruise = distance_m / max(1e-6, cruise_ms)
    E_cruise = P_cruise * t_cruise / 3600.0
    return E_hover + E_cruise, P_hover, P_cruise, t_cruise

# “Lidar” rays for perception toy
def lidar_scan(vehicle_xy: Tuple[float,float], obstacles: np.ndarray, r_max=20.0, num_rays=180, res=0.5):
    vx, vy = vehicle_xy
    angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)
    hits = []
    for th in angles:
        d = 0.0
        hit = r_max
        while d <= r_max:
            x = vx + d * math.cos(th)
            y = vy + d * math.sin(th)
            if np.any(np.hypot(obstacles[:,0]-x, obstacles[:,1]-y) < 0.6):
                hit = d
                break
            d += res
        hits.append(hit)
    return np.array(hits), angles

def draw_lidar(vehicle_xy, obs, hits, angles, r_max, safety_radius):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(obs[:,0], obs[:,1], marker="s")
    vx, vy = vehicle_xy
    ax.scatter([vx],[vy], marker="o")
    for d,th in zip(hits, angles):
        x = vx + d * math.cos(th)
        y = vy + d * math.sin(th)
        ax.plot([vx,x],[vy,y], linewidth=0.6)
    safety = plt.Circle((vx,vy), safety_radius, fill=False, linestyle="--")
    ax.add_patch(safety)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Perception Sandbox (toy lidar)")
    ax.set_xlabel("m"); ax.set_ylabel("m"); ax.grid(True, alpha=0.2)
    return fig, ax

# Health score toy model (0–100)
def health_score(hours, cycles, max_temp_c, vib_g_rms):
    h = clamp(100 - (hours*0.8 + cycles*0.3 + max(0, max_temp_c-40)*1.2 + vib_g_rms*8), 0, 100)
    due = h < 60 or max_temp_c > 75 or vib_g_rms > 1.2
    return h, due

# ─────────────────────────────────────────────────────────
# ONE set of tabs (fix for repeated UI)
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["City Hop Planner", "Perception Sandbox", "Health Monitor"])

# ─────────────────────────────────────────────────────────
# 1) City Hop Planner
# ─────────────────────────────────────────────────────────
with tab1:
    st.subheader("City Hop Planner (Autonomy + UTM lite)")

    cols = st.columns(4)
    grid_W = cols[0].slider("Grid width", 20, 60, 40, help="Cells across (coarse city grid).")
    grid_H = cols[1].slider("Grid height", 20, 60, 40)
    scale_m = cols[2].selectbox("Meters per cell", [25, 50, 75, 100], index=1)
    reserve_pct = cols[3].slider("Battery reserve %", 0, 30, 15)

    cols = st.columns(4)
    mass_kg = cols[0].number_input("Gross mass (kg)", 300.0, 5000.0, 1100.0, step=50.0)
    cruise_ms = cols[1].number_input("Cruise speed (m/s)", 20.0, 120.0, 45.0, step=5.0)
    hover_time_s = cols[2].number_input("Total hover time (s)", 0.0, 300.0, 60.0, step=5.0, help="Takeoff + landing + holds")
    battery_Wh = cols[3].number_input("Usable battery (Wh)", 50000.0, 400000.0, 120000.0, step=5000.0)

    st.markdown("**Pads (grid coords)**: (0,0) bottom-left; avoid no-fly rectangles.")
    cols = st.columns(4)
    start = (int(cols[0].number_input("Start Y", 0, grid_H-1, 5)),
             int(cols[1].number_input("Start X", 0, grid_W-1, 5)))
    goal  = (int(cols[2].number_input("Goal Y", 0, grid_H-1, grid_H-6)),
             int(cols[3].number_input("Goal X", 0, grid_W-1, grid_W-6)))

    st.divider()
    st.markdown("**No-Fly Zones (NFZ)** – rectangles on the grid (inclusive).")
    cols = st.columns(5)
    nfz_count = cols[0].slider("How many NFZs?", 0, 5, 2)
    nfz_rects: List[Tuple[int,int,int,int]] = []
    for i in range(nfz_count):
        with st.expander(f"NFZ #{i+1}", expanded=(i<2)):
            c = st.columns(4)
            y0 = int(c[0].number_input("y0", 0, grid_H-1, 12 + 3*i, key=f"y0{i}"))
            x0 = int(c[1].number_input("x0", 0, grid_W-1, 10 + 3*i, key=f"x0{i}"))
            y1 = int(c[2].number_input("y1", 0, grid_H-1, 18 + 3*i, key=f"y1{i}"))
            x1 = int(c[3].number_input("x1", 0, grid_W-1, 18 + 3*i, key=f"x1{i}"))
            y0,y1 = sorted([y0,y1]); x0,x1 = sorted([x0,x1])
            nfz_rects.append((y0,x0,y1,x1))

    # Build grid
    grid = np.zeros((grid_H, grid_W), dtype=int)
    for (y0,x0,y1,x1) in nfz_rects:
        grid[y0:y1+1, x0:x1+1] = 1  # blocked

    # Pathfind
    path = astar(grid, start, goal)
    path_len_cells = len(path)-1 if path else None
    distance_m = (path_len_cells or 0) * scale_m

    # Energy + GO/NO-GO
    E_wh, P_hover, P_cruise, t_cruise = energy_wh(distance_m, cruise_ms, mass_kg, hover_time_s)
    reserve = (reserve_pct/100.0) * battery_Wh
    go = (path is not None) and (E_wh + reserve <= battery_Wh)

    # Plot
    fig, ax = plt.subplots(figsize=(5.5,5.5))
    draw_grid_path(grid, path, start, goal, nfz_rects, scale_m, ax=ax, title="Auto-Route with A*")
    st.pyplot(fig, use_container_width=False)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Path distance", f"{distance_m/1000:.2f} km")
    c2.metric("Hover power (toy)", f"{P_hover/1000:.1f} kW")
    c3.metric("Cruise power (toy)", f"{P_cruise/1000:.1f} kW")
    c4.metric("Cruise time", f"{t_cruise/60:.1f} min")

    st.info(f"Energy needed (toy): **{E_wh:,.0f} Wh** | Battery: **{battery_Wh:,.0f} Wh** | Reserve: **{reserve:,.0f} Wh**")
    st.success("GO: route found and energy within limits ✅") if go else st.error("NO-GO: blocked or not enough energy ❌")

# ─────────────────────────────────────────────────────────
# 2) Perception Sandbox
# ─────────────────────────────────────────────────────────
with tab2:
    st.subheader("Perception Sandbox (toy lidar)")

    cols = st.columns(4)
    num_obs = cols[0].slider("Obstacles", 2, 25, 8)
    field_m = cols[1].selectbox("Field size (m)", [30, 40, 60], index=1)
    r_max = cols[2].slider("Lidar range (m)", 5, 40, 20)
    safety_radius = cols[3].slider("Safety stop radius (m)", 1, 10, 5)

    seed = st.slider("Random seed", 0, 999, 7)
    rng = np.random.default_rng(seed)
    obs = rng.uniform(low=-field_m/2, high=field_m/2, size=(num_obs, 2))
    vehicle_xy = (0.0, 0.0)

    hits, ang = lidar_scan(vehicle_xy, obs, r_max=r_max, num_rays=180, res=0.25)
    nearest = hits.min() if len(hits) else r_max
    brake = nearest < safety_radius

    fig, _ = draw_lidar(vehicle_xy, obs, hits, ang, r_max, safety_radius)
    st.pyplot(fig)

    c1,c2,c3 = st.columns(3)
    c1.metric("Nearest obstacle", f"{nearest:.2f} m")
    c2.metric("Rays with hit", f"{np.sum(hits<r_max)}")
    c3.metric("Brake/avoid", "YES" if brake else "NO")
    st.caption("Concept: basic perception → nearest-obstacle check → automatic safety stop/avoid.")

# ─────────────────────────────────────────────────────────
# 3) Health Monitor
# ─────────────────────────────────────────────────────────
with tab3:
    st.subheader("Predictive Maintenance (toy)")

    cols = st.columns(4)
    hours = cols[0].number_input("Flight hours", 0.0, 5000.0, 820.0, step=10.0)
    cycles = cols[1].number_input("Charge cycles", 0.0, 6000.0, 950.0, step=10.0)
    temp = cols[2].number_input("Max pack temp (°C)", 0.0, 120.0, 68.0, step=1.0)
    vib = cols[3].number_input("Vibration gRMS", 0.0, 5.0, 0.7, step=0.1)

    score, due = health_score(hours, cycles, temp, vib)
    st.metric("Health score", f"{score:.0f} / 100")
    st.write("• Maintenance window: **Soon**" if due else "• Maintenance window: **Not yet due**")
    st.caption("Concept: combine hours/cycles/thermal/vibration into a quick health index for scheduling.")

st.divider()
st.markdown(
    "**Mapping to your slide:**  \n"
    "• *Real-time autonomy*: A* route around NFZs + GO/NO-GO energy.  \n"
    "• *Perception*: 2D lidar toy → nearest-obstacle brake/avoid.  \n"
    "• *Predictive maintenance*: quick health score.  \n"
    "• *UTM/fleet mgmt*: extend with time slots for multi-aircraft later."
)
