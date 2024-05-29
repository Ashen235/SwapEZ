from flask import Flask, request, jsonify
from flask_cors import CORS
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
import networkx as nx

app = Flask(__name__)
CORS(app)

# In-memory graph to manage nodes and edges
graph = nx.Graph()

# Quantum simulator backend
simulator = Aer.get_backend('aer_simulator')

@app.route('/add_endpoint', methods=['POST'])
def add_endpoint():
    data = request.json
    endpoint_id = data['id']
    if graph.has_node(endpoint_id):
        return jsonify({"error": "Node already exists"}), 400
    graph.add_node(endpoint_id, type='endpoint')
    return jsonify({"status": "endpoint added", "id": endpoint_id}), 200

@app.route('/add_repeater', methods=['POST'])
def add_repeater():
    data = request.json
    repeater_id = data['id']
    if graph.has_node(repeater_id):
        return jsonify({"error": "Node already exists"}), 400
    graph.add_node(repeater_id, type='repeater')
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
        return jsonify({"error": "Edge length does not satisfy the triangle inequality"}), 400

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
        return jsonify({"error": "New edge length does not satisfy the triangle inequality"}), 400

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
    data = request.json
    node1 = data['node1']
    node2 = data['node2']
    
    if not graph.has_node(node1) or not graph.has_node(node2):
        return jsonify({"error": "One or both nodes do not exist"}), 400

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    
    transpiled_qc = transpile(qc, simulator)
    job = simulator.run(transpiled_qc, shots=1)
    result = job.result()
    counts = result.get_counts(transpiled_qc)
    
    graph.nodes[node1]['epr_partner'] = node2
    graph.nodes[node2]['epr_partner'] = node1
    
    return jsonify({"status": "EPR pair created", "nodes": [node1, node2], "result": counts}), 200

@app.route('/measure', methods=['POST'])
def measure():
    data = request.json
    node = data['node']
    if 'qubit' in graph.nodes[node]:
        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)  # Measure qubit 0 into classical bit 0
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
    
    if 'epr_partner' not in graph.nodes[sender] or graph.nodes[sender]['epr_partner'] != receiver:
        return jsonify({"error": "no EPR pair between nodes"}), 400
    
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.h(1)
    qc.measure([0, 1], [0, 1])  # Measure qubits 0 and 1 into classical bits 0 and 1
    qc.cx(1, 2)
    qc.cz(0, 2)
    qc.measure(2, 2)  # Measure qubit 2 into classical bit 2
    
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
    
    if 'epr_partner' not in graph.nodes[sender] or graph.nodes[sender]['epr_partner'] != receiver:
        return jsonify({"error": "no EPR pair between nodes"}), 400
    
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
    qc.measure([0, 1], [0, 1])  # Measure qubits 0 and 1 into classical bits 0 and 1
    
    transpiled_qc = transpile(qc, simulator)
    job = simulator.run(transpiled_qc, shots=1)
    result = job.result()
    counts = result.get_counts(transpiled_qc)
    
    return jsonify({"status": "superdense coding performed", "result": counts}), 200

@app.route('/request_entanglement', methods=['POST'])
def request_entanglement():
    data = request.json
    endpoint1 = data['endpoint1']
    endpoint2 = data['endpoint2']
    
    if not graph.has_node(endpoint1) or not graph.has_node(endpoint2):
        return jsonify({"error": "One or both endpoints do not exist"}), 400

    try:
        shortest_path = nx.shortest_path(graph, source=endpoint1, target=endpoint2, weight='cost')
        epr_pairs = []
        entanglement_swaps = []

        for i in range(len(shortest_path) - 1):
            node1 = shortest_path[i]
            node2 = shortest_path[i + 1]
            epr_pairs.append((node1, node2))
            if i > 0 and i < len(shortest_path) - 2:
                entanglement_swaps.append((shortest_path[i], shortest_path[i + 1]))

        return jsonify({
            "status": "entanglement path computed",
            "path": shortest_path,
            "epr_pairs": epr_pairs,
            "entanglement_swaps": entanglement_swaps
        }), 200
    except nx.NetworkXNoPath:
        return jsonify({"error": "No path found between the endpoints"}), 400

def is_edge_valid(node1, node2, cost):
    try:
        shortest_path_length = nx.shortest_path_length(graph, source=node1, target=node2, weight='cost')
        return cost <= shortest_path_length
    except nx.NetworkXNoPath:
        return True  # If no path exists, the edge is valid by default

if __name__ == '__main__':
    app.run(debug=True)
