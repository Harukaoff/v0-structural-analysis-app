"""
Generate Stress Diagrams and Visualizations
Creates deformation diagram, shear force diagram, and bending moment diagram
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import base64
from io import BytesIO

def plot_deformed_shape(nodes, elements, displacements, scale=1000):
    """
    Plot the deformed shape of the structure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot original structure
    for element in elements:
        if element['type'] != 'beam':
            continue
        
        node1_id = element['node_ids'][0]
        node2_id = element['node_ids'][1]
        
        node1 = next(n for n in nodes if n['id'] == node1_id)
        node2 = next(n for n in nodes if n['id'] == node2_id)
        
        ax.plot([node1['x'], node2['x']], 
                [node1['y'], node2['y']], 
                'k--', linewidth=1, alpha=0.3, label='元の形状' if element['id'] == 0 else '')
    
    # Plot deformed structure
    for element in elements:
        if element['type'] != 'beam':
            continue
        
        node1_id = element['node_ids'][0]
        node2_id = element['node_ids'][1]
        
        node1 = next(n for n in nodes if n['id'] == node1_id)
        node2 = next(n for n in nodes if n['id'] == node2_id)
        
        disp1 = next(d for d in displacements if d['node_id'] == node1_id)
        disp2 = next(d for d in displacements if d['node_id'] == node2_id)
        
        # Apply scaled displacements
        x1_def = node1['x'] + disp1['u'] * scale
        y1_def = node1['y'] + disp1['v'] * scale
        x2_def = node2['x'] + disp2['u'] * scale
        y2_def = node2['y'] + disp2['v'] * scale
        
        ax.plot([x1_def, x2_def], 
                [y1_def, y2_def], 
                'b-', linewidth=2, label='変形後の形状' if element['id'] == 0 else '')
    
    # Plot nodes
    for node in nodes:
        ax.plot(node['x'], node['y'], 'ko', markersize=6)
        ax.text(node['x'], node['y'] + 0.02, f"N{node['id']}", 
                ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('X座標 (m)', fontsize=12)
    ax.set_ylabel('Y座標 (m)', fontsize=12)
    ax.set_title(f'変形図 (変形スケール: {scale}倍)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(loc='best')
    
    # Convert to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{img_base64}"

def plot_shear_force_diagram(nodes, elements, element_forces):
    """
    Plot shear force diagram for all elements
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot structure outline
    for element in elements:
        if element['type'] != 'beam':
            continue
        
        node1_id = element['node_ids'][0]
        node2_id = element['node_ids'][1]
        
        node1 = next(n for n in nodes if n['id'] == node1_id)
        node2 = next(n for n in nodes if n['id'] == node2_id)
        
        ax.plot([node1['x'], node2['x']], 
                [node1['y'], node2['y']], 
                'k-', linewidth=1, alpha=0.3)
    
    # Plot shear force diagram
    offset_scale = 0.0001  # Scale for visual offset
    
    for force in element_forces:
        element = next(e for e in elements if e['id'] == force['element_id'])
        node1_id = element['node_ids'][0]
        node2_id = element['node_ids'][1]
        
        node1 = next(n for n in nodes if n['id'] == node1_id)
        node2 = next(n for n in nodes if n['id'] == node2_id)
        
        # Calculate perpendicular direction
        dx = node2['x'] - node1['x']
        dy = node2['y'] - node1['y']
        L = np.sqrt(dx**2 + dy**2)
        
        if L == 0:
            continue
        
        # Perpendicular unit vector
        perp_x = -dy / L
        perp_y = dx / L
        
        # Shear force values
        V1 = force['V1'] * offset_scale
        V2 = force['V2'] * offset_scale
        
        # Plot shear force polygon
        x_points = [node1['x'], 
                    node1['x'] + V1 * perp_x, 
                    node2['x'] + V2 * perp_x, 
                    node2['x']]
        y_points = [node1['y'], 
                    node1['y'] + V1 * perp_y, 
                    node2['y'] + V2 * perp_y, 
                    node2['y']]
        
        ax.fill(x_points, y_points, color='red', alpha=0.3, edgecolor='red', linewidth=2)
        
        # Add value labels
        mid_x = (node1['x'] + node2['x']) / 2
        mid_y = (node1['y'] + node2['y']) / 2
        ax.text(mid_x, mid_y, f"{force['V1']:.1f}N", 
                fontsize=8, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('X座標 (m)', fontsize=12)
    ax.set_ylabel('Y座標 (m)', fontsize=12)
    ax.set_title('せん断力図', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Convert to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{img_base64}"

def plot_bending_moment_diagram(nodes, elements, element_forces):
    """
    Plot bending moment diagram for all elements
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot structure outline
    for element in elements:
        if element['type'] != 'beam':
            continue
        
        node1_id = element['node_ids'][0]
        node2_id = element['node_ids'][1]
        
        node1 = next(n for n in nodes if n['id'] == node1_id)
        node2 = next(n for n in nodes if n['id'] == node2_id)
        
        ax.plot([node1['x'], node2['x']], 
                [node1['y'], node2['y']], 
                'k-', linewidth=1, alpha=0.3)
    
    # Plot bending moment diagram
    offset_scale = 0.00005  # Scale for visual offset
    
    for force in element_forces:
        element = next(e for e in elements if e['id'] == force['element_id'])
        node1_id = element['node_ids'][0]
        node2_id = element['node_ids'][1]
        
        node1 = next(n for n in nodes if n['id'] == node1_id)
        node2 = next(n for n in nodes if n['id'] == node2_id)
        
        # Calculate perpendicular direction
        dx = node2['x'] - node1['x']
        dy = node2['y'] - node1['y']
        L = np.sqrt(dx**2 + dy**2)
        
        if L == 0:
            continue
        
        # Perpendicular unit vector
        perp_x = -dy / L
        perp_y = dx / L
        
        # Bending moment values
        M1 = force['M1'] * offset_scale
        M2 = force['M2'] * offset_scale
        
        # Create smooth curve for moment diagram
        n_points = 20
        t = np.linspace(0, 1, n_points)
        
        # Quadratic interpolation for moment
        M_curve = M1 * (1 - t) + M2 * t
        
        x_curve = node1['x'] * (1 - t) + node2['x'] * t
        y_curve = node1['y'] * (1 - t) + node2['y'] * t
        
        x_offset = x_curve + M_curve * perp_x
        y_offset = y_curve + M_curve * perp_y
        
        # Plot moment diagram
        x_points = np.concatenate([[node1['x']], x_offset, [node2['x']]])
        y_points = np.concatenate([[node1['y']], y_offset, [node2['y']]])
        
        ax.fill(x_points, y_points, color='blue', alpha=0.3, edgecolor='blue', linewidth=2)
        
        # Add value labels
        mid_x = (node1['x'] + node2['x']) / 2
        mid_y = (node1['y'] + node2['y']) / 2
        ax.text(mid_x, mid_y, f"M1:{force['M1']:.1f}\nM2:{force['M2']:.1f}", 
                fontsize=7, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('X座標 (m)', fontsize=12)
    ax.set_ylabel('Y座標 (m)', fontsize=12)
    ax.set_title('曲げモーメント図', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Convert to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{img_base64}"

def generate_all_diagrams(normalized_result, analysis_result):
    """
    Generate all stress diagrams
    """
    nodes = normalized_result.get('nodes', [])
    elements = [e for e in normalized_result.get('elements', []) if e['type'] == 'beam']
    
    displacements = analysis_result.get('displacements', [])
    element_forces = analysis_result.get('element_forces', [])
    
    if not nodes or not elements or not displacements:
        return {"error": "Insufficient data for diagram generation"}
    
    # Convert pixel coordinates to meters
    scale = 0.01
    for node in nodes:
        node['x'] *= scale
        node['y'] *= scale
    
    # Generate diagrams
    deformation_diagram = plot_deformed_shape(nodes, elements, displacements)
    shear_diagram = plot_shear_force_diagram(nodes, elements, element_forces)
    moment_diagram = plot_bending_moment_diagram(nodes, elements, element_forces)
    
    return {
        "success": True,
        "deformation_diagram": deformation_diagram,
        "shear_diagram": shear_diagram,
        "moment_diagram": moment_diagram
    }

if __name__ == "__main__":
    # Read input from stdin
    input_data = sys.stdin.read()
    data = json.loads(input_data)
    
    normalized_result = data.get("normalized_result")
    analysis_result = data.get("analysis_result")
    
    if not normalized_result or not analysis_result:
        print(json.dumps({"error": "Missing required data"}))
        sys.exit(1)
    
    # Generate diagrams
    result = generate_all_diagrams(normalized_result, analysis_result)
    
    # Output result
    print(json.dumps(result))
