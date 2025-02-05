from flask import Flask, request, jsonify
from flask_cors import CORS
import networkx as nx
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import state_fidelity
from qiskit_aer.noise import NoiseModel, amplitude_damping_error, phase_damping_error
import math
import time
import random

app = Flask(__name__)
CORS(app)

# Global network graph: nodes represent endpoints/repeaters,
# edges represent quantum channels with an associated cost.
graph = nx.Graph()

# Constants
F0 = 0.99       # Baseline fidelity for an entangled pair
alpha = 0.007   # Probability constant for transmission 
beta = 0.003    # Decay constant for transmission 
gamma = 1.3     # Superlinear decay constant
SWAP_SUCCESS_PROB = 0.8
SWAP_EFFICIENCY = 0.99    # Fidelity penalty for swapping

# Qiskit simulators:
# - 'simulator' for noisy simulation (density matrix mode when noise is provided)
# - 'state_simulator' for ideal simulation (statevector mode)
simulator = AerSimulator()
state_simulator = AerSimulator(method="statevector")

# # Create a noise model.
# noise_model = NoiseModel()
# # Build a 1-qubit error (for one-qubit gates like 'id' and 'h')
# error_1q = amplitude_damping_error(0.01).compose(phase_damping_error(0.01))
# noise_model.add_all_qubit_quantum_error(error_1q, ['id', 'h'])
# # Build a 2-qubit error (for the 'cx' gate)
# error_2q_amp = amplitude_damping_error(0.01).tensor(amplitude_damping_error(0.01))
# error_2q_phase = phase_damping_error(0.01).tensor(phase_damping_error(0.01))
# error_2q = error_2q_amp.compose(error_2q_phase)
# noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

# def current_time():
#     return time.time()

# def apply_decoherence(fidelity, creation_time):
#     """Apply an exponential decay to the fidelity based on elapsed time."""
#     t_elapsed = current_time() - creation_time
#     decay = math.exp(-t_elapsed / T_COHERENCE)
#     return fidelity * decay

# def generate_bell_state():
#     """
#     Generate an ideal Bell state and simulate it with the noise model.
#     For the ideal (noise-free) simulation, we add save_statevector().
#     For the noisy simulation, we add save_density_matrix() and retrieve it from the result.
#     The fidelity is computed between the ideal statevector and the noisy density matrix.
#     """
#     qc = QuantumCircuit(2)
#     qc.h(0)
#     qc.cx(0, 1)
    
#     # Ideal (noise-free) simulation: save the statevector.
#     ideal_qc = qc.copy()
#     ideal_qc.save_statevector()
#     ideal_qc = transpile(ideal_qc, state_simulator)
#     ideal_job = state_simulator.run(ideal_qc, shots=1)
#     ideal_state = ideal_job.result().get_statevector()
    
#     # Noisy simulation: save the density matrix.
#     noisy_qc = qc.copy()
#     noisy_qc.save_density_matrix()
#     noisy_qc = transpile(noisy_qc, simulator)
#     noisy_job = simulator.run(noisy_qc, noise_model=noise_model, shots=1)
#     # Retrieve the density matrix from the result data.
#     noisy_state = noisy_job.result().data(0)['density_matrix']
    
#     fid = state_fidelity(ideal_state, noisy_state)
#     return fid

# def perform_entanglement_swapping(fidelity1, fidelity2):
#     """Combine the fidelities of two entangled links using a swapping operation."""
#     return fidelity1 * fidelity2 * SWAP_EFFICIENCY

@app.route('/add_endpoint', methods=['POST'])
def add_endpoint():
    data = request.json
    endpoint_id = data['id']
    if graph.has_node(endpoint_id):
        return jsonify({"error": "Node already exists"}), 400
    graph.add_node(endpoint_id, type='endpoint', epr=None)
    return jsonify({"status": "endpoint added", "id": endpoint_id}), 200

@app.route('/add_repeater', methods=['POST'])
def add_repeater():
    data = request.json
    repeater_id = data['id']
    if graph.has_node(repeater_id):
        return jsonify({"error": "Node already exists"}), 400
    graph.add_node(repeater_id, type='repeater', epr=None)
    return jsonify({"status": "repeater added", "id": repeater_id}), 200

@app.route('/add_edge', methods=['POST'])
def add_edge():
    data = request.json
    node1 = data['node1']
    node2 = data['node2']
    cost = data['cost']
    
    if not graph.has_node(node1) or not graph.has_node(node2):
        return jsonify({"error": "One or both nodes do not exist"}), 400

    if graph.has_edge(node1, node2):
        return jsonify({"error": "Edge already exists"}), 400

    if not is_edge_valid(node1, node2, cost):
        return jsonify({"error": "Edge cost does not satisfy the triangle inequality"}), 400

    graph.add_edge(node1, node2, cost=cost)
    return jsonify({"status": "edge added", "nodes": [node1, node2], "cost": cost}), 200

@app.route('/modify_edge', methods=['POST'])
def modify_edge():
    data = request.json
    node1 = data['node1']
    node2 = data['node2']
    new_cost = data['new_cost']
    
    if not graph.has_node(node1) or not graph.has_node(node2):
        return jsonify({"error": "One or both nodes do not exist"}), 400

    if not graph.has_edge(node1, node2):
        return jsonify({"error": "Edge does not exist"}), 400

    if not is_edge_valid(node1, node2, new_cost):
        return jsonify({"error": "New edge cost does not satisfy the triangle inequality"}), 400

    graph[node1][node2]['cost'] = new_cost
    return jsonify({"status": "edge modified", "nodes": [node1, node2], "new_cost": new_cost}), 200

@app.route('/remove_node', methods=['POST'])
def remove_node():
    data = request.json
    node_id = data['id']
    if not graph.has_node(node_id):
        return jsonify({"error": "Node does not exist"}), 400
    graph.remove_node(node_id)
    return jsonify({"status": "node removed", "id": node_id}), 200

@app.route('/remove_edge', methods=['POST'])
def remove_edge():
    data = request.json
    node1 = data['node1']
    node2 = data['node2']
    if not graph.has_edge(node1, node2):
        return jsonify({"error": "Edge does not exist"}), 400
    graph.remove_edge(node1, node2)
    return jsonify({"status": "edge removed", "nodes": [node1, node2]}), 200

# @app.route('/create_epr', methods=['POST'])
# def create_epr():
#     """
#     Generate an EPR pair between two nodes and record the fidelity and timestamp.
#     """
#     data = request.json
#     node1 = data['node1']
#     node2 = data['node2']
    
#     if not graph.has_node(node1) or not graph.has_node(node2):
#         return jsonify({"error": "One or both nodes do not exist"}), 400

#     fidelity = generate_bell_state()
#     creation_time = current_time()
#     fidelity = apply_decoherence(fidelity, creation_time)

#     graph.nodes[node1]['epr'] = {"partner": node2, "fidelity": fidelity, "creation_time": creation_time}
#     graph.nodes[node2]['epr'] = {"partner": node1, "fidelity": fidelity, "creation_time": creation_time}

#     return jsonify({
#         "status": "EPR pair created",
#         "nodes": [node1, node2],
#         "fidelity": fidelity,
#         "creation_time": creation_time
#     }), 200

@app.route('/measure', methods=['POST'])
def measure():
    data = request.json
    node = data['node']
    if 'epr' in graph.nodes[node] and graph.nodes[node]['epr'] is not None:
        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        transpiled_qc = transpile(qc, simulator)
        job = simulator.run(transpiled_qc, shots=1)
        result = job.result()
        counts = result.get_counts(transpiled_qc)
        return jsonify({"status": "measured", "result": counts})
    else:
        return jsonify({"error": "no qubit found in node"}), 400

@app.route('/teleport', methods=['POST'])
def teleport():
    data = request.json
    sender = data['sender']
    receiver = data['receiver']
    
    if 'epr' not in graph.nodes[sender] or graph.nodes[sender]['epr'] is None or graph.nodes[sender]['epr']['partner'] != receiver:
        return jsonify({"error": "No EPR pair between nodes"}), 400
    
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.h(1)
    qc.measure([0, 1], [0, 1])
    qc.cx(1, 2)
    qc.cz(0, 2)
    qc.measure(2, 2)
    
    transpiled_qc = transpile(qc, simulator)
    job = simulator.run(transpiled_qc, shots=1)
    result = job.result()
    counts = result.get_counts(transpiled_qc)
    
    return jsonify({"status": "teleportation performed", "result": counts}), 200

@app.route('/superdense_coding', methods=['POST'])
def superdense_coding():
    data = request.json
    sender = data['sender']
    receiver = data['receiver']
    message = data['message']
    
    if 'epr' not in graph.nodes[sender] or graph.nodes[sender]['epr'] is None or graph.nodes[sender]['epr']['partner'] != receiver:
        return jsonify({"error": "No EPR pair between nodes"}), 400
    
    qc = QuantumCircuit(2, 2)
    if message == '00':
        pass
    elif message == '01':
        qc.x(0)
    elif message == '10':
        qc.z(0)
    elif message == '11':
        qc.x(0)
        qc.z(0)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure([0, 1], [0, 1])
    
    transpiled_qc = transpile(qc, simulator)
    job = simulator.run(transpiled_qc, shots=1)
    result = job.result()
    counts = result.get_counts(transpiled_qc)
    
    return jsonify({"status": "superdense coding performed", "result": counts}), 200

def current_time():
    return time.time()


def transmission_probability(distance):
    """Success probability of entanglement generation over an edge based on distance."""
    return math.exp(-alpha * (distance ** gamma))

def fidelity_loss(distance):
    """Fidelity decay function for entanglement transmission over an edge."""
    return math.exp(-beta * (distance ** gamma))

def effective_edge_cost(u, v, edge_data):
    """Modified cost function for pathfinding based on probability and fidelity."""
    dist = edge_data["cost"]
    
    prob = transmission_probability(dist)
    fidelity = fidelity_loss(dist)

    if prob * fidelity > 0:
        return -math.log(prob * fidelity)  # Convert to log-space for Dijkstra
    return float("inf")  # Unreachable if zero


def find_optimal_path(graph, start, end):
    """Finds the path that maximizes entanglement success using a modified cost function."""
    for edge in list(graph.edges):
        print(edge)
        print(graph[edge[0]][edge[1]])
        print(effective_edge_cost(edge[0],edge[1],graph[edge[0]][edge[1]]))
    try:
        path = nx.shortest_path(graph, source=start, target=end, weight=effective_edge_cost)
        return path
    except nx.NetworkXNoPath:
        return None

def generate_link(node1, node2, operations):
    """
    Attempts to generate an entanglement link between node1 and node2.
    Retries until successful.
    Logs all failed attempts.
    """
    
    # print(nx.to_dict_of_dicts(graph))
    # print(node1, node2)
    # print(graph[node1][node2])

    distance = graph[node1][node2].get("cost", 1)
    link_success_prob = transmission_probability(distance)
    link_fidelity = fidelity_loss(distance)

    success = False
    while not success:  # Retry until successful
        if random.random() <= link_success_prob:
            success = True
            operations.append({
                "type": "link_generation",
                "nodes": [node1, node2],
                "fidelity": link_fidelity,
                "success_probability": link_success_prob,
                "status": "success"
            })
            return link_fidelity
        else:
            operations.append({
                "type": "link_generation",
                "nodes": [node1, node2],
                "status": "failed",
                "retrying": True
            })

def generate_epr_pair():
    """Generate an EPR pair using Qiskit."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    transpiled_qc = transpile(qc, simulator)
    job = simulator.run(transpiled_qc)
    return job.result()

def perform_swap(circuit, qubit1, qubit2):
    """Perform a quantum swap operation on two qubits."""
    circuit.cx(qubit1, qubit2)
    circuit.h(qubit1)
    return circuit

def compute_final_fidelity(operations):
    """
    Computes the final fidelity of the entangled state by considering:
    - The fidelity of each initial entanglement link.
    - The fidelity of each swap performed along the chain.
    - The correct propagation of fidelity along the path.
    """
    gen_fidelities = {}
    final_fidelity = None

    # Step 1: Gather initial fidelities from link generation operations
    for op in operations:
        if op["type"] == "link_generation" and op["status"] == "success":
            node1, node2 = op["nodes"]
            fidelity = op["fidelity"]
            gen_fidelities[(node1, node2)] = fidelity
            gen_fidelities[(node2, node1)] = fidelity  # Store both directions

    # Step 2: Process swaps and propagate fidelity
    for op in operations:
        if op["type"] == "swap" and op["status"] == "success":
            node = op["node"]
            neighbor1, fidelity1 = op["inputs"][0]  # First input link fidelity
            neighbor2, fidelity2 = op["inputs"][1]  # Second input link fidelity

            # Compute fidelity after swap using the correct formula
            F_swap = (fidelity1 * fidelity2) / (1 + (1 - fidelity1) * (1 - fidelity2))

            # Update fidelity mapping after swap
            gen_fidelities[(neighbor1, neighbor2)] = F_swap
            gen_fidelities[(neighbor2, neighbor1)] = F_swap

            # Keep track of the last computed fidelity
            final_fidelity = F_swap

    # If no swaps occurred, fallback to the highest generation fidelity
    if final_fidelity is None:
        final_fidelity = max(gen_fidelities.values(), default=0)

    return final_fidelity

def getIndexOfTuple(l, index, value):
    for pos,t in enumerate(l):
        if t[index] == value:
            return pos

    # Matches behavior of list.index
    raise ValueError("list.index(x): x not in list")

def generate_links(path, memory, operations, start = 0, end = -1):
    if end <= start and end != -1: 
        return jsonify({"error": "End parameter is less or equal than start. Cannot generate links!"}), 400

    end = (len(path) - 1) if end == -1 else end

    for i in range(start, end):
        node1, node2 = path[i], path[i + 1]
        link_fidelity = generate_link(node1, node2, operations)
        memory[node1].append((node2, link_fidelity))
        memory[node2].append((node1, link_fidelity))

def do_swaps(path, memory, operations, start = 0, end = -1):
    if (end - start) < 2 and end != -1: 
            return jsonify({"error": "Start -> End length is less than 2 links. Cannot do swaps!"}), 400

    end = (len(path) - 1) if end == -1 else end

    for i in range(start + 1, end):
        node = path[i]

        swap_success = False
        while not swap_success:

            neighbor1, fidelity1 = memory[node].pop()
            neighbor2, fidelity2 = memory[node].pop()

            if random.random() <= SWAP_SUCCESS_PROB:
                swap_success = True
                # Apply swap efficiency **directly here**
                new_fidelity = fidelity1 * fidelity2 * SWAP_EFFICIENCY
                # print("neighbor1", neighbor1, memory[neighbor1], [n if n != node else neighbor2 for n, _ in memory[neighbor1]])
                # print("neighbor2", neighbor2, memory[neighbor2], [n if n != node else neighbor1 for n, _ in memory[neighbor2]])

                memory[neighbor1] = [(n if n != node else neighbor2, new_fidelity) for n, _ in memory[neighbor1]]
                memory[neighbor2] = [(n if n != node else neighbor1, new_fidelity) for n, _ in memory[neighbor2]]
                operations.append({
                    "type": "swap",
                    "node": node,
                    "inputs": [(neighbor1, fidelity1), (neighbor2, fidelity2)],
                    "resulting_fidelity": new_fidelity,
                    "status": "success"
                })
                # qc = perform_swap(qc, path.index(neighbor1), path.index(neighbor2))
                # print(memory, " after swap success")
            else:
                operations.append({
                    "type": "swap", 
                    "node": node,
                    "inputs": [(neighbor1, fidelity1), (neighbor2, fidelity2)],
                    "status": "failed",
                    "retrying": True
                })

                # if swap failed, remove the qubits from the memories of the end nodes of the links
                # (meaning neighbors of the -- qubits used in -- swap node)
                print("Memory on fail at node ", node, memory)
                memory[neighbor1].pop(getIndexOfTuple(memory[neighbor1], 0, node))
                memory[neighbor2].pop(getIndexOfTuple(memory[neighbor2], 0, node))
                print("Memory after removing neighbor qubits at node ", node, memory)   

                # get indexes of ends of swap operation and current node
                index1, index2 = path.index(neighbor1, start, end + 1), path.index(neighbor2, start, end + 1)
                _start, _end = (index1, index2) if index1 < index2 else (index2, index1)
                node_index = path.index(node, start + 1, end)

                # regenerate lost links
                generate_links(path, memory, operations, _start, _end)
                print("Memory after regenerating links at node ", node, memory)

                # retry swaps up to this point divide et impera style
                if node_index - _start >= 2:
                    do_swaps(path, memory, operations, _start, node_index)
                    print("Memory after redoing swaps between indexes of start and node ", start, node_index, memory)

                if _end - node_index >= 2:
                    do_swaps(path, memory, operations, _start, node_index)
                    print("Memory after redoing swaps between indexes of node and end ", node, end, memory)


@app.route('/request_entanglement', methods=['POST'])
def request_entanglement():
    """
    Computes entanglement distribution using Qiskit simulation.
    - Regenerates links in case of swap failure.
    - Tracks fidelity properly and computes the final fidelity directly from memory.
    """
    try:
        data = request.json
        endpoint1 = data['endpoint1']
        endpoint2 = data['endpoint2']

        if not graph.has_node(endpoint1) or not graph.has_node(endpoint2):
            return jsonify({"error": "One or both endpoints do not exist"}), 400

        # Find optimal path
        path = find_optimal_path(graph, endpoint1, endpoint2)
        if not path:
            print(graph)
            return jsonify({"error": "No valid entanglement path found"}), 400

        # Initialize circuit for simulation
        num_nodes = len(path)
        qc = QuantumCircuit(num_nodes)

        # Initialize memory and operations
        memory = {node: [] for node in path}
        operations = []

        # Generate entanglement links
        generate_links(path, memory, operations)
    
        # Do swaps at intermediary nodes
        do_swaps(path, memory, operations)

        # Final fidelity is now simply **the fidelity of the final entangled link**
        final_fidelity = None
        for neighbor, fidelity in memory[endpoint1]:
            if neighbor == endpoint2:
                final_fidelity = fidelity
                break

        if final_fidelity is None:
            return jsonify({"error": "Final entangled state was not established", "memory": memory, "operations": operations}), 500

        # Final Qiskit simulation
        # qc.measure_all()
        # transpiled_qc = transpile(qc, simulator)
        # job = simulator.run(transpiled_qc, shots=1024)
        # final_result = job.result()

        return jsonify({
            "status": "success",
            "path": path,
            "operations": operations,
            "final_fidelity": final_fidelity,
            "memory": memory
            # "quantum_simulation": final_result.get_counts()
        }), 200

    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(traceback_str)
        return jsonify({
            "error": "Internal server error", 
            "details": traceback_str, 
            "operations": operations,
            "memory": memory
        }), 500

@app.route('/clear_network', methods=['DELETE'])
def clear_network():
    """
    Clear the entire network configuration.
    """
    graph.clear()  # Clear all nodes and edges
    return jsonify({"status": "network cleared"}), 200

@app.route('/import_network', methods=['POST'])
def import_network():
    """
    Import a network configuration. The request JSON should have the keys "nodes" and "edges".
    This endpoint first clears the existing network configuration.
    """
    data = request.json
    print(data)
    # Clear any previously loaded configuration.
    graph.clear()
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    # Add nodes.
    for node in nodes:
        node_id = node.get("id")
        node_type = node.get("type")
        # You can add extra parameters (like positions) if needed.
        graph.add_node(node_id, type=node_type)
    # Add edges.
    for edge in edges:
        node1 = (edge.get("source")).get("id")
        node2 = (edge.get("target")).get("id")
        print(node1, node2)
        cost = edge.get("value")
        if graph.has_node(node1) and graph.has_node(node2):
            graph.add_edge(node1, node2, cost=cost)
    return jsonify({
        "status": "network imported",
        "nodeCount": graph.number_of_nodes(),
        "edgeCount": graph.number_of_edges()
    }), 200


def is_edge_valid(node1, node2, cost):
    """
    Ensure the new edge does not violate the triangle inequality.
    If no path exists, the edge is valid by default.
    """
    try:
        current_length = nx.shortest_path_length(graph, source=node1, target=node2, weight='cost')
        return cost <= current_length
    except nx.NetworkXNoPath:
        return True

if __name__ == '__main__':
    app.run(debug=True)
