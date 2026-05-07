#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import dimod
from dimod import ConstrainedQuadraticModel, Real, Integer
from dwave.system import LeapHybridCQMSampler


@dataclass
class Instance:
    n: int
    depot_ready: float
    depot_due: float
    capacity: float
    demand: np.ndarray
    ready: np.ndarray
    due: np.ndarray
    service: np.ndarray
    D: np.ndarray
    coords: np.ndarray  # (n+1,2) with depot at index 0

def load_instance(csv_path: str, capacity: int) -> Instance:
    df = pd.read_csv(csv_path, encoding='utf-8-sig').rename(columns={
        'CUST NO.':'id','XCOORD.':'x','YCOORD.':'y','DEMAND':'demand',
        'READY TIME':'ready_time','DUE DATE':'due_time','SERVICE TIME':'service_time'
    })
    depot = df.iloc[0]
    customers = df.iloc[1:].reset_index(drop=True)

    X = customers['x'].astype(float).values
    Y = customers['y'].astype(float).values
    demand = customers['demand'].astype(float).values
    ready = customers['ready_time'].astype(float).values
    due   = customers['due_time'].astype(float).values
    service = customers['service_time'].astype(float).values

    coords = np.vstack(([depot['x'], depot['y']], np.column_stack((X, Y)))).astype(float)
    n = len(customers)
    D = np.zeros((n+1, n+1))
    for i in range(n+1):
        for j in range(n+1):
            dx = coords[i,0]-coords[j,0]
            dy = coords[i,1]-coords[j,1]
            D[i,j] = math.hypot(dx, dy)

    depot_ready=float(depot['ready_time'])
    depot_due=float(depot['due_time']) if not pd.isna(depot['due_time']) else float(np.max(due)+np.max(D))

    return Instance(
        n=n,
        depot_ready=depot_ready,
        depot_due=depot_due,
        capacity=float(capacity),
        demand=demand,
        ready=ready,
        due=due,
        service=service,
        D=D,
        coords=coords
    )

def route_feasible_and_cost(inst: Instance, route: List[int]) -> Tuple[bool, float]:
    if not route:
        return True, 0.0
    t = inst.depot_ready
    dist = 0.0
    prev = 0
    for cust in route:
        dist += inst.D[prev, cust]
        t += inst.D[prev, cust]
        r = inst.ready[cust-1]; d = inst.due[cust-1]; s = inst.service[cust-1]
        if t < r:
            t = r
        if t > d:
            return False, math.inf
        t += s
        prev = cust
    dist += inst.D[prev, 0]
    t += inst.D[prev, 0]
    if t > inst.depot_due:
        return False, math.inf
    return True, dist

def split_greedy(inst: Instance, perm: List[int]) -> Tuple[List[List[int]], float, bool]:
    routes = []
    cur = []
    load = 0.0
    for idx in perm:
        d = inst.demand[idx-1]
        if load + d <= inst.capacity:
            test = cur + [idx]
            feas, _ = route_feasible_and_cost(inst, test)
            if feas:
                cur = test
                load += d
                continue
        if cur:
            feas, _ = route_feasible_and_cost(inst, cur)
            if not feas:
                return [], math.inf, False
            routes.append(cur)
        cur = [idx]
        load = d
        feas, _ = route_feasible_and_cost(inst, cur)
        if not feas:
            return [], math.inf, False
    if cur:
        feas, _ = route_feasible_and_cost(inst, cur)
        if not feas:
            return [], math.inf, False
        routes.append(cur)
    total = 0.0
    for r in routes:
        _, c = route_feasible_and_cost(inst, r)
        total += c
    return routes, total, True

def init_population(n: int, size: int) -> List[List[int]]:
    base = list(range(1, n+1))
    pop = []
    for _ in range(size):
        p = base[:]
        random.shuffle(p)
        pop.append(p)
    return pop

def ox_crossover(p1: List[int], p2: List[int]) -> List[int]:
    m = len(p1)
    a,b = sorted(random.sample(range(m),2))
    child = [-1]*m
    child[a:b+1] = p1[a:b+1]
    fill = [g for g in p2 if g not in child]
    j=0
    for i in range(m):
        if child[i] == -1:
            child[i] = fill[j]; j+=1
    return child

def swap_mutation(p: List[int], rate: float=0.2) -> None:
    if random.random() < rate:
        i,j = random.sample(range(len(p)),2)
        p[i], p[j] = p[j], p[i]

def evaluate(inst: Instance, perm: List[int]) -> Tuple[float, List[List[int]]]:
    routes, dist, feas = split_greedy(inst, perm)
    if not feas:
        return math.inf, None
    return dist, routes

def tournament(pop: List[List[int]], fit, k: int=3) -> List[int]:
    cand = random.sample(range(len(pop)), k)
    best = min(cand, key=lambda i: fit[i][0])
    return pop[best][:]

def build_vrptw_cqm(inst: Instance, node_list: List[int]):
    nodes = node_list[:]
    m = len(nodes)
    idx = {nodes[t]: t for t in range(m)}

    D = inst.D[np.ix_(nodes, nodes)]
    service = np.zeros(m)
    ready = np.zeros(m)
    due   = np.zeros(m)
    for orig, t in idx.items():
        if orig == 0:
            service[t] = 0.0
            ready[t] = inst.depot_ready
            due[t]   = inst.depot_due
        else:
            service[t] = inst.service[orig-1]
            ready[t]   = inst.ready[orig-1]
            due[t]     = inst.due[orig-1]

    allowed = [[True]*m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i == j:
                allowed[i][j] = False
                continue
            earliest = ready[i] + service[i] + D[i,j]
            if earliest > due[j] + 1e-9:
                allowed[i][j] = False

    cqm = ConstrainedQuadraticModel()

    A = {}
    for i in range(m):
        lb = float(ready[i]); ub = float(due[i])
        if ub < lb:
            ub = lb + 1.0
        A[i] = Real(f"A_{i}", lower_bound=lb, upper_bound=ub)

    y_labels: Dict[Tuple[int,int], str] = {}
    for i in range(m):
        for j in range(m):
            if i == j or not allowed[i][j]:
                continue
            lab = f"y_{i}_{j}"
            y_labels[(i,j)] = lab
            cqm.add_variable(dimod.BINARY, lab)

    U = {}
    U[0] = Integer(f"U_0", lower_bound=0, upper_bound=m-1)
    cqm.add_constraint(U[0] == 0, label="mtz_dep")
    for i in range(1, m):
        U[i] = Integer(f"U_{i}", lower_bound=1, upper_bound=m-1)

    for i in range(m):
        out_vars = [y_labels[(i,j)] for j in range(m) if (i,j) in y_labels]
        in_vars  = [y_labels[(j,i)] for j in range(m) if (j,i) in y_labels]
        if len(out_vars) == 0:
            cand = [(D[i,j], j) for j in range(m) if j != i]
            cand.sort()
            j = cand[0][1]
            lab = f"y_{i}_{j}"
            if (i,j) not in y_labels:
                y_labels[(i,j)] = lab
                cqm.add_variable(dimod.BINARY, lab)
            out_vars = [lab]
        if len(in_vars) == 0:
            cand = [(D[j,i], j) for j in range(m) if j != i]
            cand.sort()
            j = cand[0][1]
            lab = f"y_{j}_{i}"
            if (j,i) not in y_labels:
                y_labels[(j,i)] = lab
                cqm.add_variable(dimod.BINARY, lab)
            in_vars = [lab]

        cqm.add_constraint(sum(dimod.Binary(v) for v in out_vars) == 1, label=f"outdeg_{i}")
        cqm.add_constraint(sum(dimod.Binary(v) for v in in_vars) == 1, label=f"indeg_{i}")

    for i in range(m):
        for j in range(m):
            if (i,j) not in y_labels:
                continue
            lab = y_labels[(i,j)]
            M_ij = max(0.0, float(due[i] + service[i] + D[i,j] - ready[j]) + 1.0)
            cqm.add_constraint(
                A[j] - A[i] - service[i] - D[i,j] + M_ij*(1 - dimod.Binary(lab)) >= 0,
                label=f"time_{i}_{j}"
            )

    for i in range(m):
        cqm.add_constraint(A[i] >= ready[i], label=f"ready_{i}")
        cqm.add_constraint(A[i] <= due[i],   label=f"due_{i}")

    for i in range(1, m):
        for j in range(1, m):
            if i == j:
                continue
            if (i,j) not in y_labels:
                continue
            lab = y_labels[(i,j)]
            cqm.add_constraint(U[i] - U[j] + (m-1)*dimod.Binary(lab) <= m-2, label=f"mtz_{i}_{j}")

    cqm.set_objective(
        sum(inst.D[nodes[i], nodes[j]] * dimod.Binary(y_labels[(i,j)]) for (i,j) in y_labels.keys())
    )

    return cqm, y_labels

def solve_route_with_cqm(inst: Instance, route: List[int], time_limit: int, label_prefix: str = "vrptw-route"):
    m = 1 + len(route)
    if m <= 3:
        feas, dist = route_feasible_and_cost(inst, route)
        return route, dist, feas

    nodes = [0] + route
    cqm, y_labels = build_vrptw_cqm(inst, nodes)

    sampler = LeapHybridCQMSampler()
    label = f"{label_prefix}-n{len(route)}"
    res = sampler.sample_cqm(cqm, time_limit=time_limit, label=label)
    sol = res.first

    feasible = bool(sol.is_feasible)

    succ = [-1]*m
    for (i,j), lab in y_labels.items():
        if sol.sample[lab] >= 0.5:
            succ[i] = j

    tour_local = [0]
    seen = {0}
    cur = 0
    while True:
        nxt = succ[cur]
        if nxt == -1 or nxt in seen:
            break
        tour_local.append(nxt)
        seen.add(nxt)
        cur = nxt
        if cur == 0:
            break

    tour_global = [nodes[t] for t in tour_local]
    customers_seq = [v for v in tour_global if v != 0]

    dist = 0.0
    prev = 0
    for c in customers_seq:
        dist += inst.D[prev, c]; prev = c
    dist += inst.D[prev, 0]

    feas_after, _ = route_feasible_and_cost(inst, customers_seq)
    return customers_seq, dist, feasible and feas_after


def refine_routes_with_cqm(inst: Instance, routes: List[List[int]], time_limit: int):
    new_routes = []
    feas_list = []
    total = 0.0
    for r in routes:
        seq, d, feas = solve_route_with_cqm(inst, r, time_limit=time_limit)
        new_routes.append(seq)
        feas_list.append(feas)
        total += d
    return new_routes, total, feas_list

def simulate_route_schedule(inst, route):
    sched = []
    t = float(inst.depot_ready)
    prev = 0
    for cust in route:
        travel = float(inst.D[prev, cust])
        arr = t + travel
        ready = float(inst.ready[cust-1])
        due   = float(inst.due[cust-1])
        svc   = float(inst.service[cust-1])
        start = arr if arr >= ready else ready
        end   = start + svc
        sched.append({
            'cust': int(cust),
            'arr':  float(arr),
            'start':float(start),
            'end':  float(end),
            'ready':float(ready),
            'due':  float(due),
            'service': float(svc)
        })
        t = end
        prev = cust
    back = t + float(inst.D[prev, 0])
    return sched, back


def build_schedules(inst, routes):
    all_sched = []
    for r in routes:
        s, back = simulate_route_schedule(inst, r)
        all_sched.append({'route': r, 'sched': s, 'back': back})
    return all_sched


def plot_routes_map(inst, routes, schedules=None, title="", save_prefix="", show=True):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter(inst.coords[0,0], inst.coords[0,1], s=120, marker='s', label='Depot')
    ax.scatter(inst.coords[1:,0], inst.coords[1:,1], s=18, alpha=0.8, label='Customers')

    for k, r in enumerate(routes):
        if not r:
            continue
        xs = [inst.coords[0,0]] + [inst.coords[i,0] for i in r] + [inst.coords[0,0]]
        ys = [inst.coords[0,1]] + [inst.coords[i,1] for i in r] + [inst.coords[0,1]]
        ax.plot(xs, ys, linewidth=2, label=f"Route {k+1}")
        if schedules is not None:
            for order, cust in enumerate(r, start=1):
                x, y = inst.coords[cust,0], inst.coords[cust,1]
                ax.text(x, y, str(order), fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

    ax.set_title(title or "VRPTW Routes (final)")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_map.png", dpi=160)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_gantt(inst, schedules, save_prefix="", show=True):
    fig, ax = plt.subplots(figsize=(10.5, 0.5 + 0.5*sum(len(s['sched']) for s in schedules)))
    y = 0
    yticks = []
    ylabels = []

    for ridx, pack in enumerate(schedules, start=1):
        sched = pack['sched']
        for item in sched:
            ax.barh(y, item['due'] - item['ready'], left=item['ready'],
                    height=0.32, alpha=0.25)
            ax.barh(y, item['end'] - item['start'], left=item['start'],
                    height=0.32, alpha=0.9)
            ax.text(item['start'], y+0.02, f"R{ridx}-C{item['cust']}",
                    fontsize=8, color='black', va='bottom')
            yticks.append(y)
            ylabels.append(f"R{ridx}-C{item['cust']}")
            y += 0.6
        ax.axhline(y-0.2, color='gray', linestyle=':', linewidth=0.8)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Time")
    ax.set_title("Time Windows vs Actual Service (Gantt)")
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_gantt.png", dpi=160)
    if show:
        plt.show()
    else:
        plt.close(fig)

def run_ga(inst: Instance,
           pop_size: int=120,
           generations: int=300,
           cx_rate: float=0.9,
           mut_rate: float=0.25,
           tournament_k: int=3,
           live_plot: bool=False) -> Tuple[float, List[List[int]], float]:
    population = init_population(inst.n, pop_size)
    fitness = [evaluate(inst, p) for p in population]
    best_idx = min(range(pop_size), key=lambda i: fitness[i][0])
    best_cost, best_routes = fitness[best_idx]

    if live_plot:
        plt.ion()
        fig, ax = plt.subplots()
        xs = [0]; ys = [best_cost if best_cost != math.inf else np.nan]
        line, = ax.plot(xs, ys, marker='o')
        ax.set_xlabel("Generation"); ax.set_ylabel("Best cost")
        ax.set_title("GA Progress (Best cost vs Generation)")
        plt.tight_layout(); fig.canvas.draw(); fig.canvas.flush_events()
        def update_plot(gen, cost):
            xs.append(gen); ys.append(cost if cost != math.inf else np.nan)
            line.set_xdata(xs); line.set_ydata(ys)
            ax.relim(); ax.autoscale_view()
            fig.canvas.draw(); fig.canvas.flush_events()
        update_plot(0, best_cost)
    else:
        update_plot = lambda gen, cost: None

    t0 = time.perf_counter()
    for g in range(1, generations+1):
        new_pop = []
        elite = population[best_idx][:]
        new_pop.append(elite)
        while len(new_pop) < pop_size:
            p1 = tournament(population, fitness, tournament_k)
            p2 = tournament(population, fitness, tournament_k)
            child = p1[:]
            if random.random() < cx_rate:
                child = ox_crossover(p1, p2)
            swap_mutation(child, mut_rate)
            new_pop.append(child)

        population = new_pop
        fitness = [evaluate(inst, p) for p in population]
        best_idx = min(range(pop_size), key=lambda i: fitness[i][0])
        if fitness[best_idx][0] < best_cost:
            best_cost, best_routes = fitness[best_idx]

        update_plot(g, best_cost)

    elapsed = time.perf_counter() - t0

    if live_plot:
        ax.annotate(f"Elapsed: {elapsed:.3f}s for {generations} gens",
                    xy=(xs[-1], ys[-1]), xytext=(0, 0), textcoords="offset points")
        plt.ioff(); plt.show()

    return best_cost, best_routes, elapsed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=str, help="Path to Solomon-style VRPTW CSV (e.g., C101.csv)")
    parser.add_argument("--capacity", type=int, default=200)
    parser.add_argument("--pop", type=int, default=120)
    parser.add_argument("--gen", type=int, default=300)
    parser.add_argument("--cx", type=float, default=0.9)
    parser.add_argument("--mut", type=float, default=0.25)
    parser.add_argument("--tk", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--live", type=int, default=0)
    parser.add_argument("--cqm_time", type=int, default=5, help="LeapHybridCQM time_limit per route (seconds)")
    parser.add_argument("--viz", type=int, default=0, choices=[0,1,2],
                        help="0:none, 1:map only, 2:map+gantt")
    parser.add_argument("--save", type=str, default="",
                        help="PNG save prefix (e.g., --save result/c101). Empty = show windows only.")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed % (2**32-1))

    inst = load_instance(args.csv, args.capacity)
    ga_cost, ga_routes, ga_elapsed = run_ga(inst,
                                            pop_size=args.pop,
                                            generations=args.gen,
                                            cx_rate=args.cx,
                                            mut_rate=args.mut,
                                            tournament_k=args.tk,
                                            live_plot=bool(args.live))

    t_ref0 = time.perf_counter()
    refined_routes, refined_cost, feas_list = refine_routes_with_cqm(inst, ga_routes, time_limit=args.cqm_time)
    t_ref1 = time.perf_counter()
    refine_elapsed = t_ref1 - t_ref0

    print(f"Best cost: {refined_cost}")
    print(f"Number of routes: {len(refined_routes)}")
    for i, r in enumerate(refined_routes, start=1):
        feas, dist = route_feasible_and_cost(inst, r)
        print(f"Route {i:02d} | stops={len(r)} | dist={dist:.3f} | feasible={feas}")

    print(f"\nElapsed time to {args.gen} generations (GA): {ga_elapsed:.3f} seconds")
    print(f"CQM refine time (total over routes): {refine_elapsed:.3f} seconds")

    if args.viz > 0:
        schedules = build_schedules(inst, refined_routes)
        title = f"GA + CQM VRPTW | cost={refined_cost:.1f} | routes={len(refined_routes)}"
        plot_routes_map(inst, refined_routes, schedules=schedules, title=title,
                        save_prefix=args.save, show=(args.save==""))
        if args.viz >= 2:
            plot_gantt(inst, schedules, save_prefix=args.save, show=(args.save==""))


if __name__ == "__main__":
    main()
