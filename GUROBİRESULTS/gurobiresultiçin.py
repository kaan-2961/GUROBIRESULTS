# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 12:11:06 2025

@author: birolduru
"""

import gurobipy as gp
from gurobipy import GRB, quicksum
import pandas as pd
import math

# --------- Parameters ---------
vehicle_count = 10
vehicle_speed = 35  # km/h
max_time = 3       # hours per vehicle route
service_time = 0.05 # hours 
fixed_cost = 50     # per vehicle 
distance_cost = 2   # per km

# --------- Input: Read Coordinates from CSV ---------
input_csv = "set_covering_osm_data.csv"  # CSV file with columns "latitude" and "longitude"
df = pd.read_csv(input_csv)
if df.empty or not {"latitude", "longitude"}.issubset(df.columns):
    raise ValueError("Input CSV must have non-empty columns 'latitude' and 'longitude'")
# Convert to a list of coordinates, with depot as the first record.
coordinates = df[['latitude', 'longitude']].values.tolist()
n = len(coordinates)

# --------- Distance Calculation (Haversine) ---------
def haversine(coord1, coord2):
    """Calculate distance (in km) between two (lat, lon) points using the Haversine formula."""
    from math import radians, sin, cos, sqrt, atan2
    R = 6371  # Earth radius in km
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    # Protect against small floating point errors
    a = min(1.0, max(0.0, a))
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Build distance and travel time matrices.
dist = [[haversine(coordinates[i], coordinates[j]) for j in range(n)] for i in range(n)]
time = [[dist[i][j] / vehicle_speed for j in range(n)] for i in range(n)]

# --------- Gurobi Model ---------
m = gp.Model("MultiVehicleTSP")

# Decision variables:
# x[i,j,k] = 1 if vehicle k goes from node i to node j.
x = m.addVars(n, n, vehicle_count, vtype=GRB.BINARY, name="x")
# u[i,k] for subtour elimination (MTZ formulation).
u = m.addVars(n, vehicle_count, vtype=GRB.CONTINUOUS, name="u")
# y[k] indicates if vehicle k is used.
y = m.addVars(vehicle_count, vtype=GRB.BINARY, name="y")

# --------- Objective: Minimize Total Cost ---------
m.setObjective(
    quicksum(y[k] * fixed_cost for k in range(vehicle_count)) +
    quicksum(x[i, j, k] * dist[i][j] * distance_cost
             for i in range(n) for j in range(n) for k in range(vehicle_count) if i != j),
    GRB.MINIMIZE
)

# --------- Constraints ---------
# 1. Each customer (nodes 1...n-1) is visited exactly once.
for j in range(1, n):
    m.addConstr(quicksum(x[i, j, k] for i in range(n) for k in range(vehicle_count) if i != j) == 1)

# 2. Flow conservation for each vehicle (non-depot nodes).
for k in range(vehicle_count):
    for h in range(1, n):
        m.addConstr(
            quicksum(x[i, h, k] for i in range(n) if i != h) ==
            quicksum(x[h, j, k] for j in range(n) if j != h)
        )

# 3. Each vehicle starts and ends at the depot (node 0).
for k in range(vehicle_count):
    m.addConstr(quicksum(x[0, j, k] for j in range(1, n)) <= y[k])
    m.addConstr(quicksum(x[i, 0, k] for i in range(1, n)) <= y[k])

# 4. Time limit for each vehicle route (travel time + service time).
for k in range(vehicle_count):
    m.addConstr(
        quicksum(x[i, j, k] * (time[i][j] + service_time)
                 for i in range(n) for j in range(n) if i != j) <= max_time
    )

# 5. Subtour elimination via lazy constraints (a simple MTZ-like approach).
def subtour_elim(model, where):
    if where == GRB.Callback.MIPSOL:
        for k in range(vehicle_count):
            vals = model.cbGetSolution([x[i, j, k] for i in range(n) for j in range(n)])
            selected = [(i, j) for idx, (i, j) in enumerate(((i, j) for i in range(n) for j in range(n))) if vals[idx] > 0.5 and i != j]

            # Use a DFS (starting at each node except depot) to detect subtours.
            visited = [False] * n
            def dfs(node, edges):
                tour = []
                stack = [node]
                while stack:
                    cur = stack.pop()
                    if not visited[cur]:
                        visited[cur] = True
                        tour.append(cur)
                        for (i, j) in edges:
                            if i == cur and not visited[j]:
                                stack.append(j)
                return tour

            for start in range(1, n):
                if not visited[start]:
                    subtour = dfs(start, selected)
                    if 0 not in subtour and len(subtour) < n:
                        model.cbLazy(
                            quicksum(x[i, j, k] for i in subtour for j in subtour if i != j) <= len(subtour) - 1
                        )

m.Params.LazyConstraints = 1
m.Params.OutputFlag = 1
m.Params.TimeLimit = 4800        
m.Params.MIPGap = 0.05

m.optimize(subtour_elim)

# --------- Print the Solution ---------
if m.SolCount > 0:
    total_cost = 0
    print("\n📦 SOLUTION DETAILS:")
    for k in range(vehicle_count):
        route = []
        visited = set()
        current_node = 0
        total_distance = 0
        total_time_k = 0

        # Build the route for vehicle k, starting at depot (node 0).
        while True:
            route.append(current_node)
            visited.add(current_node)
            next_node = None
            for j in range(n):
                if j != current_node and x[current_node, j, k].X > 0.5:
                    next_node = j
                    total_distance += dist[current_node][j]
                    total_time_k += time[current_node][j] + service_time
                    break
            if next_node is None or next_node in visited:
                break
            current_node = next_node

        # Ensure the vehicle returns to the depot.
        if route[-1] != 0:
            route.append(0)
            total_distance += dist[current_node][0]
            total_time_k += time[current_node][0] + service_time

        if len(route) > 1:
            distance_cost_k = total_distance * distance_cost
            total_cost_k = distance_cost_k + fixed_cost
            total_cost += total_cost_k

            print(f"\n🚚 Vehicle {k+1} Route: {' -> '.join(str(node) for node in route)}")
            print(f"  ⏱ Total time: {total_time_k:.2f} hours")
            print(f"  📏 Total distance: {total_distance:.2f} km")
            print(f"  💵 Total cost: {total_cost_k:.2f} $ (distance: {distance_cost_k:.2f} $ + fixed: {fixed_cost:.2f} $)")
        else:
            print(f"\n🚚 Vehicle {k+1} not used.")

    print(f"\n🔢 Total system cost: {total_cost:.2f} $")
else:
    print("No solution found.")
