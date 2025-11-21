"""
Stiffness Matrix Method for Structural Analysis
Performs structural analysis on beam structures with various supports and loads
Calculates displacements, reactions, shear forces, and bending moments
"""

import sys
import json
import numpy as np
from scipy.linalg import solve

# Material properties (default values for steel)
E = 200e9  # Young's modulus (Pa) - 200 GPa for steel
I = 1e-5   # Moment of inertia (m^4) - default value

class StructuralAnalyzer:
    def __init__(self, nodes, elements, supports, loads, material_props=None):
        """
        Initialize structural analyzer
        
        Args:
            nodes: List of node dictionaries with id, x, y
            elements: List of beam elements connecting nodes
            supports: List of support conditions
            loads: List of applied loads
            material_props: Dictionary with E (Young's modulus) and I (moment of inertia)
        """
        self.nodes = sorted(nodes, key=lambda n: n['id'])
        self.elements = elements
        self.supports = supports
        self.loads = loads
        
        if material_props:
            self.E = material_props.get('E', E)
            self.I = material_props.get('I', I)
        else:
            self.E = E
            self.I = I
        
        self.n_nodes = len(self.nodes)
        self.n_dofs = self.n_nodes * 3  # 3 DOFs per node: u, v, theta
        
        # Create node id to index mapping
        self.node_map = {node['id']: i for i, node in enumerate(self.nodes)}
        
    def get_element_stiffness_matrix(self, element):
        """
        Calculate local stiffness matrix for a beam element
        
        Returns 6x6 stiffness matrix for 2D beam with axial, shear, and bending
        """
        node1_idx = self.node_map[element['node_ids'][0]]
        node2_idx = self.node_map[element['node_ids'][1]]
        
        node1 = self.nodes[node1_idx]
        node2 = self.nodes[node2_idx]
        
        # Element length
        dx = node2['x'] - node1['x']
        dy = node2['y'] - node1['y']
        L = np.sqrt(dx**2 + dy**2)
        
        if L == 0:
            return np.zeros((6, 6))
        
        # Direction cosines
        c = dx / L
        s = dy / L
        
        # Area (assumed constant)
        A = 0.01  # m^2
        
        # Local stiffness matrix coefficients
        EA_L = (self.E * A) / L
        EI_L = (self.E * self.I) / L
        EI_L2 = (self.E * self.I) / (L**2)
        EI_L3 = (self.E * self.I) / (L**3)
        
        # Local stiffness matrix
        k_local = np.array([
            [EA_L,      0,           0,          -EA_L,     0,           0         ],
            [0,         12*EI_L3,    6*EI_L2,    0,         -12*EI_L3,   6*EI_L2   ],
            [0,         6*EI_L2,     4*EI_L,     0,         -6*EI_L2,    2*EI_L    ],
            [-EA_L,     0,           0,          EA_L,      0,           0         ],
            [0,         -12*EI_L3,   -6*EI_L2,   0,         12*EI_L3,    -6*EI_L2  ],
            [0,         6*EI_L2,     2*EI_L,     0,         -6*EI_L2,    4*EI_L    ]
        ])
        
        # Transformation matrix
        T = np.array([
            [c,  s,  0,  0,  0,  0],
            [-s, c,  0,  0,  0,  0],
            [0,  0,  1,  0,  0,  0],
            [0,  0,  0,  c,  s,  0],
            [0,  0,  0,  -s, c,  0],
            [0,  0,  0,  0,  0,  1]
        ])
        
        # Global stiffness matrix
        k_global = T.T @ k_local @ T
        
        return k_global, [node1_idx, node2_idx]
    
    def assemble_global_stiffness_matrix(self):
        """Assemble global stiffness matrix"""
        K = np.zeros((self.n_dofs, self.n_dofs))
        
        for element in self.elements:
            if element['type'] != 'beam':
                continue
            
            k_elem, node_indices = self.get_element_stiffness_matrix(element)
            
            # Assembly
            for i in range(2):  # 2 nodes per element
                for j in range(2):
                    node_i = node_indices[i]
                    node_j = node_indices[j]
                    
                    for di in range(3):  # 3 DOFs per node
                        for dj in range(3):
                            global_i = node_i * 3 + di
                            global_j = node_j * 3 + dj
                            local_i = i * 3 + di
                            local_j = j * 3 + dj
                            
                            K[global_i, global_j] += k_elem[local_i, local_j]
        
        return K
    
    def assemble_load_vector(self):
        """Assemble global load vector"""
        F = np.zeros(self.n_dofs)
        
        for load in self.loads:
            load_type = load['type']
            
            if load_type == 'load':  # Point load
                # Find nearest node or beam
                if 'node_id' in load:
                    node_idx = self.node_map[load['node_id']]
                    # Assume vertical downward load
                    force_magnitude = load.get('magnitude', 1000)  # Default 1kN
                    F[node_idx * 3 + 1] -= force_magnitude  # Negative for downward
                    
            elif load_type == 'UDL':  # Uniformly distributed load
                if 'connected_beam_id' in load:
                    # Find the beam element
                    beam = next((e for e in self.elements if e['id'] == load['connected_beam_id']), None)
                    if beam:
                        # Distribute load to nodes
                        node1_idx = self.node_map[beam['node_ids'][0]]
                        node2_idx = self.node_map[beam['node_ids'][1]]
                        
                        node1 = self.nodes[node1_idx]
                        node2 = self.nodes[node2_idx]
                        L = np.sqrt((node2['x'] - node1['x'])**2 + (node2['y'] - node1['y'])**2)
                        
                        w = load.get('magnitude', 100)  # N/m
                        # Equivalent nodal loads for UDL
                        F[node1_idx * 3 + 1] -= w * L / 2
                        F[node2_idx * 3 + 1] -= w * L / 2
                        F[node1_idx * 3 + 2] -= w * L**2 / 12  # Moment
                        F[node2_idx * 3 + 2] += w * L**2 / 12  # Moment
                        
            elif load_type in ['momentL', 'momentR']:  # Applied moment
                if 'node_id' in load:
                    node_idx = self.node_map[load['node_id']]
                    moment_magnitude = load.get('magnitude', 100)  # N⋅m
                    if load_type == 'momentL':
                        F[node_idx * 3 + 2] += moment_magnitude
                    else:
                        F[node_idx * 3 + 2] -= moment_magnitude
        
        return F
    
    def apply_boundary_conditions(self, K, F):
        """Apply support boundary conditions"""
        # Create list of constrained DOFs
        constrained_dofs = []
        
        for support in self.supports:
            support_type = support['type']
            node_id = support.get('node_id')
            
            if node_id is None:
                continue
            
            node_idx = self.node_map[node_id]
            
            if support_type == 'fixed':
                # Fixed: all DOFs constrained
                constrained_dofs.extend([node_idx * 3, node_idx * 3 + 1, node_idx * 3 + 2])
            elif support_type == 'pin':
                # Pinned: u and v constrained, theta free
                constrained_dofs.extend([node_idx * 3, node_idx * 3 + 1])
            elif support_type == 'roller':
                # Roller: v constrained, u and theta free
                constrained_dofs.append(node_idx * 3 + 1)
            elif support_type == 'hinge':
                # Hinge: moment released (theta free), but u and v constrained
                constrained_dofs.extend([node_idx * 3, node_idx * 3 + 1])
        
        # Remove duplicates and sort
        constrained_dofs = sorted(list(set(constrained_dofs)))
        
        # Free DOFs
        all_dofs = list(range(self.n_dofs))
        free_dofs = [dof for dof in all_dofs if dof not in constrained_dofs]
        
        return constrained_dofs, free_dofs
    
    def solve(self):
        """
        Solve the structural analysis problem
        Returns displacements, reactions, and internal forces
        """
        # Assemble global stiffness matrix and load vector
        K = self.assemble_global_stiffness_matrix()
        F = self.assemble_load_vector()
        
        # Apply boundary conditions
        constrained_dofs, free_dofs = self.apply_boundary_conditions(K, F)
        
        if len(free_dofs) == 0:
            return {
                "error": "Structure is overconstrained - no free DOFs",
                "displacements": [],
                "reactions": []
            }
        
        # Partition matrices
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        F_f = F[free_dofs]
        
        # Solve for displacements
        try:
            d_f = solve(K_ff, F_f)
        except np.linalg.LinAlgError:
            return {
                "error": "Singular stiffness matrix - structure is unstable",
                "displacements": [],
                "reactions": []
            }
        
        # Full displacement vector
        d = np.zeros(self.n_dofs)
        d[free_dofs] = d_f
        
        # Calculate reactions
        R = K @ d - F
        
        # Extract nodal displacements and reactions
        displacements = []
        reactions = []
        
        for i, node in enumerate(self.nodes):
            disp = {
                "node_id": node['id'],
                "u": float(d[i * 3]),
                "v": float(d[i * 3 + 1]),
                "theta": float(d[i * 3 + 2])
            }
            displacements.append(disp)
            
            react = {
                "node_id": node['id'],
                "Rx": float(R[i * 3]),
                "Ry": float(R[i * 3 + 1]),
                "M": float(R[i * 3 + 2])
            }
            reactions.append(react)
        
        # Calculate element forces
        element_forces = self.calculate_element_forces(d)
        
        return {
            "success": True,
            "displacements": displacements,
            "reactions": reactions,
            "element_forces": element_forces
        }
    
    def calculate_element_forces(self, d):
        """Calculate internal forces in each element"""
        element_forces = []
        
        for element in self.elements:
            if element['type'] != 'beam':
                continue
            
            node1_idx = self.node_map[element['node_ids'][0]]
            node2_idx = self.node_map[element['node_ids'][1]]
            
            node1 = self.nodes[node1_idx]
            node2 = self.nodes[node2_idx]
            
            # Element displacement vector
            d_elem = np.array([
                d[node1_idx * 3],
                d[node1_idx * 3 + 1],
                d[node1_idx * 3 + 2],
                d[node2_idx * 3],
                d[node2_idx * 3 + 1],
                d[node2_idx * 3 + 2]
            ])
            
            # Element geometry
            dx = node2['x'] - node1['x']
            dy = node2['y'] - node1['y']
            L = np.sqrt(dx**2 + dy**2)
            
            if L == 0:
                continue
            
            c = dx / L
            s = dy / L
            
            # Local forces = k_local * T * d_elem
            k_elem, _ = self.get_element_stiffness_matrix(element)
            
            T = np.array([
                [c,  s,  0,  0,  0,  0],
                [-s, c,  0,  0,  0,  0],
                [0,  0,  1,  0,  0,  0],
                [0,  0,  0,  c,  s,  0],
                [0,  0,  0,  -s, c,  0],
                [0,  0,  0,  0,  0,  1]
            ])
            
            d_local = T @ d_elem
            
            # Approximate internal forces
            # Shear at start and end
            V1 = 12 * self.E * self.I / L**3 * (d_local[1] - d_local[4]) + \
                 6 * self.E * self.I / L**2 * (d_local[2] + d_local[5])
            V2 = -V1
            
            # Moments at start and end
            M1 = 6 * self.E * self.I / L**2 * (d_local[1] - d_local[4]) + \
                 4 * self.E * self.I / L * d_local[2] + \
                 2 * self.E * self.I / L * d_local[5]
            M2 = 6 * self.E * self.I / L**2 * (d_local[1] - d_local[4]) + \
                 2 * self.E * self.I / L * d_local[2] + \
                 4 * self.E * self.I / L * d_local[5]
            
            element_forces.append({
                "element_id": element['id'],
                "node1_id": element['node_ids'][0],
                "node2_id": element['node_ids'][1],
                "V1": float(V1),
                "V2": float(V2),
                "M1": float(M1),
                "M2": float(M2),
                "length": float(L)
            })
        
        return element_forces

def prepare_analysis_data(normalized_result):
    """Prepare data from normalized result for structural analysis"""
    nodes = normalized_result.get('nodes', [])
    elements = [e for e in normalized_result.get('elements', []) if e['type'] == 'beam']
    supports = [e for e in normalized_result.get('elements', []) if e['type'] in ['pin', 'roller', 'fixed', 'hinge']]
    loads = [e for e in normalized_result.get('elements', []) if e['type'] in ['load', 'UDL', 'momentL', 'momentR']]
    
    # Convert pixel coordinates to meters (scale factor)
    scale = 0.01  # 1 pixel = 0.01 m
    
    for node in nodes:
        node['x'] *= scale
        node['y'] *= scale
    
    for element in elements:
        element['width'] *= scale
        element['center']['x'] *= scale
        element['center']['y'] *= scale
    
    return nodes, elements, supports, loads

if __name__ == "__main__":
    # Read input from stdin
    input_data = sys.stdin.read()
    data = json.loads(input_data)
    
    normalized_result = data.get("normalized_result")
    
    if not normalized_result:
        print(json.dumps({"error": "No normalized result provided"}))
        sys.exit(1)
    
    # Prepare analysis data
    nodes, elements, supports, loads = prepare_analysis_data(normalized_result)
    
    if not nodes or not elements:
        print(json.dumps({"error": "Insufficient structural data"}))
        sys.exit(1)
    
    # Create analyzer and solve
    analyzer = StructuralAnalyzer(nodes, elements, supports, loads)
    result = analyzer.solve()
    
    # Output result
    print(json.dumps(result))
