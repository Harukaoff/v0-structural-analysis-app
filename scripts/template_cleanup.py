"""
Template Matching and Cleanup Script
Cleans up detected structural elements by:
- Snapping angles to 15-degree increments
- Aligning support Y-coordinates
- Connecting nearby beams with supports and loads
- Applying templates for standardized drawing
"""

import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import base64
from io import BytesIO

# Template paths
TEMPLATE_DIR = r"C:\Users\morim\Downloads\graduation\templates"
TEMPLATES = {
    "hinge": "hinge.png",
    "pin": "pin.png",
    "roller": "roller.png",
    "fixed": "fixed.png",
    "UDL": "UDL.png",
    "load": "load.png",
    "beam": "beam.png",
    "momentR": "momentR.png",
    "momentL": "momentL.png"
}

def snap_angle_to_15(angle):
    """Snap angle to nearest 15-degree increment"""
    return round(angle / 15) * 15

def align_support_y_coordinates(elements, threshold=20):
    """Align Y-coordinates of supports that are close to each other"""
    supports = [e for e in elements if e['type'] in ['pin', 'roller', 'fixed', 'hinge']]
    
    if not supports:
        return elements
    
    # Group supports by similar Y-coordinates
    y_groups = []
    for support in supports:
        y = support['center']['y']
        added = False
        
        for group in y_groups:
            if abs(y - np.mean([s['center']['y'] for s in group])) < threshold:
                group.append(support)
                added = True
                break
        
        if not added:
            y_groups.append([support])
    
    # Align each group to average Y
    for group in y_groups:
        avg_y = np.mean([s['center']['y'] for s in group])
        for support in group:
            support['center']['y'] = avg_y
    
    return elements

def connect_elements(elements, distance_threshold=30):
    """
    Connect nearby beams with supports and loads
    Returns list of connections (node graph)
    """
    beams = [e for e in elements if e['type'] == 'beam']
    supports = [e for e in elements if e['type'] in ['pin', 'roller', 'fixed', 'hinge']]
    loads = [e for e in elements if e['type'] in ['load', 'UDL', 'momentL', 'momentR']]
    
    nodes = []
    connections = []
    node_id = 0
    
    # Create nodes from beam endpoints and centers
    for beam in beams:
        # Calculate beam endpoints based on center, angle, and length
        angle_rad = np.radians(beam['angle'])
        half_length = beam['width'] / 2
        
        x1 = beam['center']['x'] - half_length * np.cos(angle_rad)
        y1 = beam['center']['y'] - half_length * np.sin(angle_rad)
        x2 = beam['center']['x'] + half_length * np.cos(angle_rad)
        y2 = beam['center']['y'] + half_length * np.sin(angle_rad)
        
        beam['endpoints'] = [{'x': x1, 'y': y1}, {'x': x2, 'y': y2}]
        beam['node_ids'] = []
        
        # Check if endpoints should merge with existing nodes
        for endpoint in beam['endpoints']:
            merged = False
            for node in nodes:
                dist = np.sqrt((node['x'] - endpoint['x'])**2 + (node['y'] - endpoint['y'])**2)
                if dist < distance_threshold:
                    beam['node_ids'].append(node['id'])
                    merged = True
                    break
            
            if not merged:
                nodes.append({'id': node_id, 'x': endpoint['x'], 'y': endpoint['y'], 'type': 'beam_end'})
                beam['node_ids'].append(node_id)
                node_id += 1
    
    # Connect supports to nearest nodes
    for support in supports:
        min_dist = float('inf')
        nearest_node_id = None
        
        for node in nodes:
            dist = np.sqrt((node['x'] - support['center']['x'])**2 + 
                          (node['y'] - support['center']['y'])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_node_id = node['id']
        
        if nearest_node_id is not None and min_dist < distance_threshold:
            support['node_id'] = nearest_node_id
            # Update node type
            for node in nodes:
                if node['id'] == nearest_node_id:
                    node['type'] = support['type']
    
    # Connect loads to nearest beams
    for load in loads:
        min_dist = float('inf')
        nearest_beam = None
        
        for beam in beams:
            # Calculate distance from load to beam line
            angle_rad = np.radians(beam['angle'])
            
            # Vector from beam center to load
            dx = load['center']['x'] - beam['center']['x']
            dy = load['center']['y'] - beam['center']['y']
            
            # Distance to beam line
            beam_vec = np.array([np.cos(angle_rad), np.sin(angle_rad)])
            point_vec = np.array([dx, dy])
            
            # Project onto beam
            projection = np.dot(point_vec, beam_vec)
            
            # Check if projection is within beam length
            if abs(projection) <= beam['width'] / 2:
                perpendicular = point_vec - projection * beam_vec
                dist = np.linalg.norm(perpendicular)
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_beam = beam
        
        if nearest_beam is not None and min_dist < distance_threshold:
            load['connected_beam_id'] = nearest_beam['id']
            load['distance_from_center'] = min_dist
    
    return elements, nodes

def normalize_elements(detection_result):
    """
    Main cleanup function: normalize detected elements
    """
    elements = detection_result.get('elements', [])
    
    if not elements:
        return detection_result
    
    # Step 1: Snap angles to 15-degree increments
    for element in elements:
        if element['type'] in ['beam', 'load', 'UDL']:
            element['angle'] = snap_angle_to_15(element['angle'])
    
    # Step 2: Align support Y-coordinates
    elements = align_support_y_coordinates(elements)
    
    # Step 3: Connect elements
    elements, nodes = connect_elements(elements)
    
    # Update result
    detection_result['elements'] = elements
    detection_result['nodes'] = nodes
    detection_result['normalized'] = True
    
    return detection_result

def draw_normalized_structure(detection_result, original_image_base64):
    """
    Draw the normalized structure using templates
    """
    # Decode original image
    if ',' in original_image_base64:
        original_image_base64 = original_image_base64.split(',')[1]
    
    image_data = base64.b64decode(original_image_base64)
    image = Image.open(BytesIO(image_data)).convert('RGBA')
    
    # Create drawing overlay
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    elements = detection_result.get('elements', [])
    
    # Draw beams first
    for element in elements:
        if element['type'] == 'beam':
            angle_rad = np.radians(element['angle'])
            half_length = element['width'] / 2
            
            x1 = element['center']['x'] - half_length * np.cos(angle_rad)
            y1 = element['center']['y'] - half_length * np.sin(angle_rad)
            x2 = element['center']['x'] + half_length * np.cos(angle_rad)
            y2 = element['center']['y'] + half_length * np.sin(angle_rad)
            
            draw.line([(x1, y1), (x2, y2)], fill=(0, 100, 255, 200), width=6)
    
    # Draw nodes
    nodes = detection_result.get('nodes', [])
    for node in nodes:
        draw.ellipse(
            [(node['x'] - 5, node['y'] - 5), (node['x'] + 5, node['y'] + 5)],
            fill=(255, 0, 0, 200),
            outline=(0, 0, 0, 255)
        )
        # Draw node ID
        draw.text((node['x'] + 8, node['y'] - 8), str(node['id']), fill=(0, 0, 0, 255))
    
    # Draw supports
    for element in elements:
        if element['type'] in ['pin', 'roller', 'fixed', 'hinge']:
            x, y = element['center']['x'], element['center']['y']
            
            # Draw support symbol (simplified)
            if element['type'] == 'pin':
                draw.ellipse([(x - 8, y - 8), (x + 8, y + 8)], fill=(0, 200, 0, 200), outline=(0, 0, 0, 255))
            elif element['type'] == 'roller':
                draw.ellipse([(x - 8, y - 8), (x + 8, y + 8)], fill=(0, 200, 200, 200), outline=(0, 0, 0, 255))
            elif element['type'] == 'fixed':
                draw.rectangle([(x - 10, y - 10), (x + 10, y + 10)], fill=(200, 0, 0, 200), outline=(0, 0, 0, 255))
            elif element['type'] == 'hinge':
                draw.ellipse([(x - 6, y - 6), (x + 6, y + 6)], fill=(200, 200, 0, 200), outline=(0, 0, 0, 255))
    
    # Draw loads
    for element in elements:
        if element['type'] in ['load', 'UDL', 'momentL', 'momentR']:
            x, y = element['center']['x'], element['center']['y']
            
            if element['type'] == 'load':
                # Point load arrow
                draw.line([(x, y - 30), (x, y)], fill=(255, 0, 0, 200), width=3)
                draw.polygon([(x, y), (x - 5, y - 10), (x + 5, y - 10)], fill=(255, 0, 0, 200))
            elif element['type'] == 'UDL':
                # Distributed load
                angle_rad = np.radians(element['angle'])
                length = element['width']
                for i in range(0, int(length), 20):
                    xi = x + (i - length/2) * np.cos(angle_rad)
                    yi = y + (i - length/2) * np.sin(angle_rad)
                    draw.line([(xi, yi - 20), (xi, yi)], fill=(255, 100, 0, 200), width=2)
    
    # Composite images
    result_image = Image.alpha_composite(image, overlay)
    
    # Convert to base64
    buffered = BytesIO()
    result_image.convert('RGB').save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

if __name__ == "__main__":
    # Read input from stdin
    input_data = sys.stdin.read()
    data = json.loads(input_data)
    
    detection_result = data.get("detection_result")
    original_image = data.get("original_image")
    
    if not detection_result:
        print(json.dumps({"error": "No detection result provided"}))
        sys.exit(1)
    
    # Normalize elements
    normalized_result = normalize_elements(detection_result)
    
    # Draw normalized structure
    if original_image:
        normalized_image = draw_normalized_structure(normalized_result, original_image)
        normalized_result['normalized_image'] = normalized_image
    
    # Output result
    print(json.dumps(normalized_result))
