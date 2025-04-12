import pandas as pd
import numpy as np
import elkai
from haversine import haversine
import random

#####################################
# Parameters (adjust as needed)
#####################################
input_csv = 'set_covering_osm_data.csv'  # Input file name
t_overall = 9         # Overall allowed time in hours
t_max_cluster = 3.0      # Maximum allowed time per cluster (in hours)
speed_km_per_hr = 35.0   # Vehicle speed in km/h
service_time_hr = 0.05   # Service time per node in hours

# Output file name will incorporate the overall time
output_csv = f'gurobi_overalltime({t_overall}).csv'

#####################################
# Helper Functions
#####################################
def compute_route_time(route, speed, service_time):
    """
    Compute the total route time (in hours) for a list of (lat, lon) pairs.
    """
    total_distance = 0.0
    for i in range(len(route) - 1):
        total_distance += haversine(route[i], route[i+1])
    travel_time = total_distance / speed
    total_service_time = service_time * len(route)
    return travel_time + total_service_time

def solve_tsp_elkai(route_nodes):
    """
    Given a list of nodes (each a (lat, lon) tuple), solve the TSP using elkai.
    Returns the order of nodes and the total distance (in km) of the tour.
    """
    n = len(route_nodes)
    if n < 2:
        return list(range(n)), 0.0
    # Build the distance matrix in km.
    dmat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dmat[i][j] = haversine(route_nodes[i], route_nodes[j])
    # Convert the distance matrix to integer values for elkai.
    dmat_int = np.round(dmat * 1000).astype(int)
    try:
        order = elkai.solve_int_matrix(dmat_int)
    except RuntimeError as e:
        print("Elkai TSP solver error:", e)
        return None, None
    # Calculate total distance using the original distances.
    total_distance = 0.0
    for i in range(len(order)-1):
        total_distance += dmat[order[i]][order[i+1]]
    return order, total_distance

def optimize_cluster(depot, cluster_nodes, speed, service_time):
    """
    Optimize the route for a cluster. The route will start at the depot,
    visit all cluster nodes (optimized using elkai), and return to the depot.
    """
    tsp_nodes = [depot] + cluster_nodes
    order, total_distance = solve_tsp_elkai(tsp_nodes)
    if order is None:
        return None, None
    # Ensure the tour starts with the depot (index 0).
    if order[0] != 0:
        depot_index = order.index(0)
        order = order[depot_index:] + order[1:depot_index+1]
    # Ensure the tour returns to the depot.
    if order[-1] != 0:
        order.append(0)
    optimized_route = [tsp_nodes[i] for i in order]
    route_time = compute_route_time(optimized_route, speed, service_time)
    return optimized_route, route_time

#####################################
# Main Routine
#####################################
def main():
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error reading {input_csv}: {e}")
        return

    if df.empty:
        print("Input CSV is empty.")
        return

    # Check if required columns exist.
    required_cols = {"latitude", "longitude"}
    if not required_cols.issubset(df.columns):
        print(f"Input CSV must have columns: {required_cols}")
        return

    # The depot is assumed to be the first row.
    depot = (df.iloc[0]['latitude'], df.iloc[0]['longitude'])
    # All other rows are candidate nodes.
    unvisited = [(row['latitude'], row['longitude']) for idx, row in df.iloc[1:].iterrows()]

    cumulative_time = 0.0
    selected_nodes = []  # To collect nodes from clusters

    # Create clusters until the overall time is reached or exceeded.
    while cumulative_time < t_overall and unvisited:
        cluster = []         # Current cluster's nodes
        current_route = [depot]  # Starting from depot

        # Build a cluster with nearest neighbor selection under t_max_cluster constraint.
        while unvisited:
            last_point = current_route[-1]
            candidate = min(unvisited, key=lambda pt: haversine(last_point, pt))
            temp_route = current_route + [candidate] + [depot]
            est_time = compute_route_time(temp_route, speed_km_per_hr, service_time_hr)
            # Always add the first candidate.
            if (est_time <= t_max_cluster) or (len(cluster) == 0):
                cluster.append(candidate)
                current_route.append(candidate)
                unvisited.remove(candidate)
            else:
                break

        if not cluster:
            break

        optimized_route, cluster_time = optimize_cluster(depot, cluster, speed_km_per_hr, service_time_hr)
        if optimized_route is None:
            # Fall back to the tentative route if optimization fails.
            cluster_time = compute_route_time(current_route + [depot], speed_km_per_hr, service_time_hr)
            optimized_route = current_route + [depot]
        cumulative_time += cluster_time
        print(f"Formed a cluster with {len(cluster)} node(s), optimized route time: {cluster_time:.2f} h, cumulative time: {cumulative_time:.2f} h")
        selected_nodes.extend(cluster)

    # Write output CSV: first row is the depot, then every unique node.
    output_data = [{'latitude': depot[0], 'longitude': depot[1]}]
    seen = set()
    for node in selected_nodes:
        if node not in seen:
            output_data.append({'latitude': node[0], 'longitude': node[1]})
            seen.add(node)
    try:
        pd.DataFrame(output_data).to_csv(output_csv, index=False)
        print(f"Output CSV written with {len(output_data)} nodes (including depot) to {output_csv}")
    except Exception as e:
        print(f"Error writing {output_csv}: {e}")

if __name__ == '__main__':
    main()
