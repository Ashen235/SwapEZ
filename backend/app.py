from flask import Flask, request, jsonify
from flask_cors import CORS
import networkx as nx
from qiskit import QuantumCircuit, transpile, execute
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import state_fidelity
from qiskit_aer.noise import NoiseModel, amplitude_damping_error, phase_damping_error
import math
import time

app = Flask(__name__)
CORS(app)

# Global network graph: nodes represent endpoints/repeaters,
# edges represent quantum channels with an associated cost.
graph = nx.Graph()

# Parameters for quantum simulation and decoherence
T_COHERENCE = 100.0  # coherence time constant (seconds)
SWAP_EFFICIENCY = 0.9  # efficiency factor for entanglement swapping

# Qiskit simulators
# Use AerSimulator from qiskit.providers.aer for simulation.
simulator = AerSimulator()  # for noisy simulation with noise model
state_simulator = AerSimulator(method="statevector")  # for ideal statevector simulation

# Create a simple noise model to simulate decoherence (for demonstration)
noise_model = NoiseModel()
# Example noise: amplitude damping and phase damping on each qubit
error_amp = amplitude_damping_error(0.01)
error_phase = phase_damping_error(0.01)
noise_model.add_all_qubit_quantum_error(error_amp.compose(error_phase), ['cx', 'id', 'h'])

def current_time():
    return time.time()

def apply_decoherence(fidelity, creation_time):
    """Apply an exponential decay to the fidelity based on elapsed time."""
    t_elapsed = current_time() - creation_time
    decay = math.exp(-t_elapsed / T_COHERENCE)
    return fidelity * decay

def generate_bell_state():
    """
    Generate an ideal Bell state and simulate it with a noise model.
    Returns an estimated fidelity (compared to the ideal state).
    """
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    
    # Transpile the circuit for the noisy simulator.
    transpiled_qc = transpile(qc, simulator)
    job = execute(transpiled_qc, simulator, noise_model=noise_model, shots=1)
    result = job.result()
    
    # For fidelity estimation, simulate the ideal state separately.
    ideal_job = execute(qc, state_simulator, shots=1)
    ideal_state = ideal_job.result().get_statevector()
    
    # Simulate noisy state using statevector simulation with noise.
    noisy_job = execute(qc, state_simulator, noise_model=noise_model, shots=1)
    noisy_state = noisy_job.result().get_statevector()
    
    fid = state_fidelity(ideal_state, noisy_state)
    return fid

def perform_entanglement_swapping(fidelity1, fidelity2):
    """
    Combine the fidelities of two entangled links using a swapping operation.
    """
    return fidelity1 * fidelity2 * SWAP_EFFICIENCY

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

@app.route('/create_epr', methods=['POST'])
def create_epr():
    """
    Generate an EPR pair between two nodes and record the fidelity and timestamp.
    """
    data = request.json
    node1 = data['node1']
    node2 = data['node2']
    
    if not graph.has_node(node1) or not graph.has_node(node2):
        return jsonify({"error": "One or both nodes do not exist"}), 400

    fidelity = generate_bell_state()
    creation_time = current_time()
    fidelity = apply_decoherence(fidelity, creation_time)

    graph.nodes[node1]['epr'] = {"partner": node2, "fidelity": fidelity, "creation_time": creation_time}
    graph.nodes[node2]['epr'] = {"partner": node1, "fidelity": fidelity, "creation_time": creation_time}

    return jsonify({
        "status": "EPR pair created",
        "nodes": [node1, node2],
        "fidelity": fidelity,
        "creation_time": creation_time
    }), 200

@app.route('/measure', methods=['POST'])
def measure():
    data = request.json
    node = data['node']
    if 'epr' in graph.nodes[node] and graph.nodes[node]['epr'] is not None:
        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        transpiled_qc = transpile(qc, simulator)
        job = execute(transpiled_qc, simulator, shots=1)
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
    job = execute(transpiled_qc, simulator, shots=1)
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
    job = execute(transpiled_qc, simulator, shots=1)
    result = job.result()
    counts = result.get_counts(transpiled_qc)
    
    return jsonify({"status": "superdense coding performed", "result": counts}), 200

@app.route('/request_entanglement', methods=['POST'])
def request_entanglement():
    """
    Compute an optimized entanglement path between two endpoints.
    The optimization leverages the observation that a repeater can generate a single EPR pair 
    and send each qubit to its two adjacent nodes.
    
    Cases handled:
      - Direct link (A-B): Generate one EPR pair.
      - Single repeater (A-R-B): The repeater generates one EPR pair and distributes qubits to A and B.
      - Multi-repeater chain (A-0-1-2-B): Only the repeaters adjacent to the endpoints generate EPR pairs.
        The inner repeater(s) perform entanglement swapping to connect the links.
    """
    data = request.json
    endpoint1 = data['endpoint1']
    endpoint2 = data['endpoint2']
    
    if not graph.has_node(endpoint1) or not graph.has_node(endpoint2):
        return jsonify({"error": "One or both endpoints do not exist"}), 400

    try:
        shortest_path = nx.shortest_path(graph, source=endpoint1, target=endpoint2, weight='cost')
    except nx.NetworkXNoPath:
        return jsonify({"error": "No path found between the endpoints"}), 400

    epr_details = []
    effective_fidelity = None

    # Case 1: Direct connection: A - B
    if len(shortest_path) == 2:
        fid = generate_bell_state()
        creation_time = current_time()
        fid = apply_decoherence(fid, creation_time)
        epr_details.append({
            "nodes": [endpoint1, endpoint2],
            "initial_fidelity": fid,
            "creation_time": creation_time,
            "note": "Direct connection"
        })
        effective_fidelity = fid

    # Case 2: Single repeater: A - R - B
    elif len(shortest_path) == 3:
        repeater = shortest_path[1]
        fid = generate_bell_state()
        creation_time = current_time()
        fid = apply_decoherence(fid, creation_time)
        epr_details.append({
            "nodes": [endpoint1, repeater],
            "initial_fidelity": fid,
            "creation_time": creation_time,
            "note": "Left EPR (Repeater to A)"
        })
        epr_details.append({
            "nodes": [repeater, endpoint2],
            "initial_fidelity": fid,
            "creation_time": creation_time,
            "note": "Right EPR (Repeater to B)"
        })
        effective_fidelity = fid

    # Case 3: Multi-repeater chain: A - 0 - ... - N - B
    else:
        # Only the repeaters adjacent to the endpoints generate EPR pairs.
        left_repeater = shortest_path[1]
        right_repeater = shortest_path[-2]
        
        fid_left = generate_bell_state()
        t_left = current_time()
        fid_left = apply_decoherence(fid_left, t_left)
        epr_details.append({
            "nodes": [endpoint1, left_repeater],
            "initial_fidelity": fid_left,
            "creation_time": t_left,
            "note": "Left EPR generation"
        })
        
        fid_right = generate_bell_state()
        t_right = current_time()
        fid_right = apply_decoherence(fid_right, t_right)
        epr_details.append({
            "nodes": [right_repeater, endpoint2],
            "initial_fidelity": fid_right,
            "creation_time": t_right,
            "note": "Right EPR generation"
        })
        
        effective_fidelity = perform_entanglement_swapping(fid_left, fid_right)
        epr_details.append({
            "nodes": [left_repeater, right_repeater],
            "initial_fidelity": effective_fidelity,
            "creation_time": max(t_left, t_right),
            "note": "Swapping at inner repeater(s)"
        })
        
        # If there are additional inner repeaters (chain length > 5), process them iteratively.
        if len(shortest_path) > 5:
            for i in range(2, len(shortest_path) - 2, 2):
                fid_mid = generate_bell_state()
                t_mid = current_time()
                fid_mid = apply_decoherence(fid_mid, t_mid)
                effective_fidelity = perform_entanglement_swapping(effective_fidelity, fid_mid)
                epr_details.append({
                    "nodes": [shortest_path[i], shortest_path[i+1]],
                    "initial_fidelity": fid_mid,
                    "creation_time": t_mid,
                    "note": "Intermediate swapping"
                })
    
    return jsonify({
        "status": "entanglement path computed",
        "path": shortest_path,
        "epr_details": epr_details,
        "effective_fidelity": effective_fidelity
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
