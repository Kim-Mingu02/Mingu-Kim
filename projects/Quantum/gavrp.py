#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# See previous cell for details.

import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def load_instance(csv_path: str, capacity: int) -> 'Instance':
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

    return Instance(
        n=n,
        depot_ready=float(depot['ready_time']),
        depot_due=float(depot['due_time']),
        capacity=float(capacity),
        demand=demand,
        ready=ready,
        due=due,
        service=service,
        D=D
    )

def route_feasible_and_cost(inst: 'Instance', route: List[int]) -> Tuple[bool, float]:
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

def split_greedy(inst: 'Instance', perm: List[int]) -> Tuple[List[List[int]], float, bool]:
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

def evaluate(inst: 'Instance', perm: List[int]):
    routes, dist, feas = split_greedy(inst, perm)
    if not feas:
        return math.inf, None
    return dist, routes

def tournament(pop: List[List[int]], fit, k: int=3) -> List[int]:
    cand = random.sample(range(len(pop)), k)
    best = min(cand, key=lambda i: fit[i][0])
    return pop[best][:]

def run_ga_with_live_plot(inst: 'Instance',
                          pop_size: int=120,
                          generations: int=300,
                          cx_rate: float=0.9,
                          mut_rate: float=0.25,
                          tournament_k: int=3):
    population = init_population(inst.n, pop_size)
    fitness = [evaluate(inst, p) for p in population]
    best_idx = min(range(pop_size), key=lambda i: fitness[i][0])
    best_cost, best_routes = fitness[best_idx]

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

    plt.ioff(); plt.show()
    return best_cost, best_routes

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
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed % (2**32-1))

    inst = load_instance(args.csv, args.capacity)
    best_cost, best_routes = run_ga_with_live_plot(inst,
                                                   pop_size=args.pop,
                                                   generations=args.gen,
                                                   cx_rate=args.cx,
                                                   mut_rate=args.mut,
                                                   tournament_k=args.tk)
    print(f"Best cost: {best_cost}")
    if best_routes is not None:
        print(f"Number of routes: {len(best_routes)}")
        for i, r in enumerate(best_routes, start=1):
            feas, dist = route_feasible_and_cost(inst, r)
            print(f"Route {i:02d} | stops={len(r)} | dist={dist:.3f} | feasible={feas}")

if __name__ == "__main__":
    main()
