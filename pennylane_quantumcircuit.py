import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Define the number of qubits
num_qubits = 5

# Create a PennyLane quantum device (simulator)
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev)
def quantum_attention_circuit():
    # Step 1: State Preparation (Encoding Queries and Keys)
    qml.Hadamard(wires=0)  # Superposition for Q_h
    qml.Hadamard(wires=1)  # Superposition for K_h

    # Step 2: Normalization (Softmax-like transformation)
    qml.RY(1.2, wires=0)  # Rotation for probability scaling
    qml.RY(1.2, wires=1)

    # Step 3: Superposition States (QK Interactions)
    qml.CNOT(wires=[0, 2])  # Interaction between Q_h and transformed Q_h
    qml.CNOT(wires=[1, 3])  # Interaction between K_h and transformed K_h

    # Step 4: Compute Entangled Attention Scores
    qml.CZ(wires=[2, 3])  # Entanglement between Q_h and K_h
    qml.CZ(wires=[0, 4])  # Additional dependency modeling

    # Step 5: Masking & Scaling (Attention Score Adjustments)
    qml.RY(0.5, wires=2)  # Scaling attention scores
    qml.RY(0.5, wires=3)

    # Measurement (Final Attention Extraction)
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]  # Expectation values

# Execute the quantum circuit
results = quantum_attention_circuit()
print("Quantum Attention Scores:", results)

# Draw the circuit
drawer = qml.draw(quantum_attention_circuit)
print(drawer())
