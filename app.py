# 🛩️ eVTOL Mini-Lab — Full App
# Features: City Hop Planner, Perception Sandbox, Health Monitor, Fleet UTM
# Author: Tareq Omrani | 2025

import math, time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="eVTOL Mini-Lab", page_icon="🛩️", layout="wide")
st.title("🛩️ eVTOL Mini-Lab")
st.caption("Autonomy • Perception • Predictive Maintenance • Fleet Coordination")

# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
def astar(grid, start, goal):
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

# ─────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "City Hop Planner", "Perception Sandbox", "Health Monitor", "Fleet UTM (multi-vehicle)"
])

# ============ 1) City Hop Planner ============
with tab1:
    st.subheader("City Hop Planner (toy A*)")
    grid_W = st.slider("Grid width",20,60,40,key="plan_w")
    grid_H = st.slider("Grid height",20,60,40,key="plan_h")
    scale_m = st.selectbox("Meters/cell",[25,50,75,100],index=1,key="plan_scale")
    start=(5,5); goal=(grid_H-6,grid_W-6)
    nfz_rects=[(10,10,15,15)]
    grid=np.zeros((grid_H,grid_W),dtype=int)
    for (y0,x0,y1,x1) in nfz_rects: grid[y0:y1+1,x0:x1+1]=1
    path=astar(grid,start,goal)
    fig, ax = draw_grid(grid,start,goal,nfz_rects,scale_m,"City Grid")
    if path:
        xs=[x for y,x in path]; ys=[y for y,x in path]
        ax.plot(xs,ys,linewidth=2)
    st.pyplot(fig)

# ============ 2) Perception Sandbox ============
with tab2:
    st.subheader("Perception Sandbox (toy lidar)")
    num_obs = st.slider("Obstacles", 2, 25, 8, key="perc_obs")
    field = st.selectbox("Field size (m)", [30, 40, 60], index=1, key="perc_field")
    r_max = st.slider("Lidar range (m)", 5, 40, 20, key="perc_rmax")
    rng = np.random.default_rng(7)
    obs = rng.uniform(low=-field/2, high=field/2, size=(num_obs, 2))
    vx, vy = 0.0, 0.0
    angles = np.linspace(0, 2*np.pi, 180, endpoint=False)
    hits = []
    for th in angles:
        d = 0.0; hit = r_max
        while d <= r_max:
            x = vx + d*math.cos(th); y = vy + d*math.sin(th)
            if np.any(np.hypot(obs[:,0]-x, obs[:,1]-y) < 0.6):
                hit = d; break
            d += 0.25
        hits.append(hit)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(obs[:,0], obs[:,1], marker="s")
    ax.scatter([vx],[vy],c="red")
    for d,th in zip(hits, angles):
        ax.plot([vx,vx+d*math.cos(th)],[vy,vy+d*math.sin(th)],linewidth=0.5)
    st.pyplot(fig)

# ============ 3) Health Monitor ============
with tab3:
    st.subheader("Predictive Maintenance")
    hours = st.number_input("Flight hours",0.0,5000.0,300.0,step=10.0,key="hm_hours")
    cycles = st.number_input("Charge cycles",0.0,6000.0,250.0,step=10.0,key="hm_cycles")
    temp = st.number_input("Max pack temp (°C)",0.0,120.0,50.0,step=1.0,key="hm_temp")
    vib = st.number_input("Vibration gRMS",0.0,5.0,0.3,step=0.1,key="hm_vib")
    score, penalty, excess_temp, due = compute_health(hours, cycles, temp, vib)
    st.session_state["health_now"] = float(score)
    st.metric("Health score", f"{score:.1f} / 100")
    st.progress(int(score))

# ============ 4) Fleet UTM (multi-vehicle) ============
with tab4:
    st.subheader("Fleet UTM (multi-vehicle, adaptive coordination)")

    grid_W = st.slider("Grid width", 20, 60, 40, key="utm_w")
    grid_H = st.slider("Grid height", 20, 60, 40, key="utm_h")
    scale_m = st.selectbox("Meters/cell", [25, 50, 75, 100], index=1, key="utm_scale")
    grid = np.zeros((grid_H, grid_W), dtype=int)

    # Fleet controls
    cols = st.columns(4)
    N = cols[0].slider("Vehicles", 2, 10, 5, key="utm_N")
    base_speed = cols[1].selectbox("Base speed (cells/step)", [1, 2], index=0, key="utm_speed")
    adaptive = cols[2].toggle("Adaptive coordination", value=True, key="utm_adapt")
    steps_max = cols[3].slider("Max steps", 50, 500, 180, key="utm_steps")

    st.markdown("**Sensor Fusion Confidence (toy)**")
    c = st.columns(3)
    conf_lidar = c[0].slider("Lidar conf", 0.0, 1.0, 0.9, 0.01, key="utm_conf_l")
    conf_radar = c[1].slider("Radar conf", 0.0, 1.0, 0.8, 0.01, key="utm_conf_r")
    conf_cam   = c[2].slider("Camera conf", 0.0, 1.0, 0.7, 0.01, key="utm_conf_c")
    fusion_conf = float(np.clip(0.5*conf_lidar + 0.3*conf_radar + 0.2*conf_cam, 0.0, 1.0))
    health_now = float(st.session_state.get("health_now", 80.0)) / 100.0

    # Starts & goals
    seed = st.slider("Random seed", 0, 999, 11, key="utm_seed")
    rng = np.random.default_rng(seed)
    free_cells = np.argwhere(grid == 0)
    rng.shuffle(free_cells)
    starts = [tuple(map(int, free_cells[i])) for i in range(N)]
    goals  = [tuple(map(int, free_cells[-(i+1)])) for i in range(N)]

    # Adaptive policy
    sep_base = 1
    sep = sep_base + (0 if not adaptive else int(np.interp(1 - min(fusion_conf, health_now), [0,1], [0,2])))
    lam_conflict = 10.0 if not adaptive else float(np.interp(1 - fusion_conf, [0,1], [4.0, 18.0]))
    lam_delay    = 0.1 if not adaptive else float(np.interp(health_now, [0,1], [0.25, 0.05]))
    lam_nfz = 50.0

    def neighbors(y, x):
        for dy, dx in [(1,0),(-1,0),(0,1),(0,-1)]:
            ny, nx = y + dy, x + dx
            if 0<=ny<grid_H and 0<=nx<grid_W and grid[ny,nx]==0:
                yield (ny, nx)

    def next_step(pos, others, goal):
        def manh(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
        best, best_cost = pos, 1e9
        for ny, nx in list(neighbors(*pos)) + [pos]:
            conflict = sum(1 for o in others if (abs(o[0]-ny)+abs(o[1]-nx)) <= sep and o!=pos)
            nfz_cross = 0 if grid[ny,nx]==0 else 1
            cost = manh((ny,nx), goal) + lam_conflict*conflict + lam_nfz*nfz_cross + lam_delay
            if cost < best_cost: best, best_cost = (ny,nx), cost
        return best

    run = st.button("Run simulation", key="utm_run")
    placeholder_plot = st.empty(); placeholder_metrics = st.empty()

    positions = starts[:]; energy_used = np.zeros(N, dtype=float)
    los_viol = 0; reached = np.zeros(N, dtype=bool)

    if run:
        for t in range(steps_max):
            new_pos=[]
            for i,p in enumerate(positions):
                if reached[i]: new_pos.append(p); continue
                others = positions[:i]+positions[i+1:]
                step = next_step(p, others, goals[i])
                new_pos.append(step)
                cell_moves = abs(step[0]-p[0])+abs(step[1]-p[1])
                energy_used[i]+=cell_moves*scale_m*0.6
                if step==goals[i]: reached[i]=True
            positions=new_pos

            for i in range(N):
                for j in range(i+1,N):
                    if (abs(positions[i][0]-positions[j][0])+abs(positions[i][1]-positions[j][1])) < sep:
                        los_viol+=1

            fig, ax = draw_grid(grid, (-1,-1), (-1,-1), [], scale_m, "Fleet UTM")
            xs=[x for y,x in positions]; ys=[y for y,x in positions]
            ax.scatter(xs,ys,s=60)
            gx=[x for y,x in goals]; gy=[y for y,x in goals]
            ax.scatter(gx,gy,marker="X",s=70)
            placeholder_plot.pyplot(fig)

            eta=np.mean([abs(p[0]-g[0])+abs(p[1]-g[1]) for p,g in zip(positions,goals)])
            done=int(np.sum(reached))
            placeholder_metrics.write(
                f"Step {t+1}/{steps_max} | Reached {done}/{N} | Mean remaining {eta:.1f} cells | "
                f"Sep buffer {sep} | LOS {los_viol} | Mean energy {np.mean(energy_used):.0f} Wh"
            )
            if done==N: break

        st.success("Simulation finished")

        # ─── Compare runs panel ───
        mean_rem_cells = float(np.mean([abs(p[0]-g[0])+abs(p[1]-g[1]) for p,g in zip(positions,goals)]))
        pct_reached = float(100.0*np.sum(reached)/N)
        mean_energy = float(np.mean(energy_used))
        run_summary = {
            "Adaptive": bool(adaptive),
            "Vehicles": int(N),
            "Grid": f"{grid_H}×{grid_W}",
            "SepBuffer(cells)": int(sep),
            "MaxSteps": int(steps_max),
            "Reached(%)": round(pct_reached, 1),
            "LOS_violations": int(los_viol),
            "MeanRemainingDist(cells)": round(mean_rem_cells, 1),
            "MeanEnergy(Wh)": int(mean_energy),
            "FusionConf": round(fusion_conf, 2),
            "Health": round(health_now, 2),
        }
        st.divider(); st.subheader("Compare runs")
        st.session_state.setdefault("utm_runs", [])
        ccmp = st.columns([1,1,2])
        if ccmp[0].button("➕ Record this run", key="utm_record"):
            st.session_state["utm_runs"].append(run_summary); st.success("Recorded.")
        if ccmp[1].button("🗑️ Clear all", key="utm_clear"):
            st.session_state["utm_runs"]=[]; st.warning("Cleared.")
        if st.session_state["utm_runs"]:
            df=pd.DataFrame(st.session_state["utm_runs"]); st.dataframe(df,use_container_width=True)
            st.markdown("**Side-by-side selection**")
            left_idx=st.number_input("Left index",0,len(df)-1,0,key="utm_left_idx")
            right_idx=st.number_input("Right index",0,len(df)-1,min(1,len(df)-1),key="utm_right_idx")
            left_row=df.iloc[int(left_idx)].to_dict(); right_row=df.iloc[int(right_idx)].to_dict()
            cL,cR=st.columns(2)
            with cL: st.markdown("##### Left"); st.json(left_row)
            with cR: st.markdown("##### Right"); st.json(right_row)
            st.info(
                f"Comparison\n\n• Left (Adaptive={left_row['Adaptive']}): "
                f"LOS={left_row['LOS_violations']}, Reached={left_row['Reached(%)']}%, Energy={left_row['MeanEnergy(Wh)']} Wh\n"
                f"• Right (Adaptive={right_row['Adaptive']}): "
                f"LOS={right_row['LOS_violations']}, Reached={right_row['Reached(%)']}%, Energy={right_row['MeanEnergy(Wh)']} Wh\n\n"
                "Lower **LOS** and similar **Reached/Energy** typically indicate better adaptive coordination."
            )
        else:
            st.caption("No runs recorded yet. Click **Record this run** after a simulation.")
