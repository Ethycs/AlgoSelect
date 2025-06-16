#!/usr/bin/env python3
"""
Convert JSON heatmap data to TikZ code for LaTeX.
Usage: python json_to_tikz.py <json_file>
"""

import json
import sys
import numpy as np

def json_to_tikz(json_file):
    """Convert JSON heatmap data to TikZ code."""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    problems = data['problems']
    algorithms = data['algorithms']
    runtimes = data['runtimes']
    min_runtime = data['min_runtime']
    max_runtime = data['max_runtime']
    
    # Generate TikZ code
    tikz_code = []
    tikz_code.append("% TikZ code for algorithm performance heatmap")
    tikz_code.append("% Generated from: " + json_file)
    tikz_code.append("\\begin{tikzpicture}[scale=0.5]")
    tikz_code.append("  % Define color mapping")
    tikz_code.append("  \\definecolor{mincolor}{RGB}{68,1,84}    % viridis purple")
    tikz_code.append("  \\definecolor{midcolor}{RGB}{33,144,140} % viridis teal")
    tikz_code.append("  \\definecolor{maxcolor}{RGB}{253,231,37} % viridis yellow")
    tikz_code.append("")
    
    # Cell size
    cell_width = 1.0
    cell_height = 0.8
    
    # Draw cells
    tikz_code.append("  % Draw heatmap cells")
    for i, problem in enumerate(problems):
        for j, algo in enumerate(algorithms):
            runtime = runtimes[i][j]
            
            if runtime < 0:  # Missing value
                color = "gray!30"
            else:
                # Normalize runtime to [0, 1]
                if max_runtime > min_runtime:
                    normalized = (runtime - min_runtime) / (max_runtime - min_runtime)
                else:
                    normalized = 0.5
                
                # Map to color (simple linear interpolation)
                # For viridis_r: low values = yellow, high values = purple
                normalized = 1 - normalized  # Reverse for viridis_r
                
                if normalized < 0.5:
                    # Interpolate between yellow and teal
                    color = f"maxcolor!{int(100*(1-2*normalized))}!midcolor"
                else:
                    # Interpolate between teal and purple
                    color = f"midcolor!{int(100*(2-2*normalized))}!mincolor"
            
            x = j * cell_width
            y = -i * cell_height  # Negative to go top-to-bottom
            
            tikz_code.append(f"  \\fill[{color}] ({x},{y}) rectangle ({x+cell_width},{y+cell_height});")
    
    # Draw grid
    tikz_code.append("")
    tikz_code.append("  % Draw grid")
    tikz_code.append("  \\draw[gray, very thin] (0,0) grid[step=%g] (%g,%g);" % 
                     (cell_width, len(algorithms)*cell_width, -len(problems)*cell_height))
    
    # Add labels
    tikz_code.append("")
    tikz_code.append("  % Algorithm labels (x-axis)")
    for j, algo in enumerate(algorithms):
        x = j * cell_width + cell_width/2
        y = 0.3
        # Shorten algorithm names for display
        short_name = algo.replace("_Algo", "").replace("Algorithm", "")[:10]
        tikz_code.append(f"  \\node[rotate=45, anchor=west, font=\\tiny] at ({x},{y}) {{{short_name}}};")
    
    tikz_code.append("")
    tikz_code.append("  % Problem labels (y-axis)")
    for i, problem in enumerate(problems):
        x = -0.3
        y = -i * cell_height - cell_height/2
        # Shorten problem names for display
        short_name = problem.replace("_Pilot", "")[:15]
        tikz_code.append(f"  \\node[anchor=east, font=\\tiny] at ({x},{y}) {{{short_name}}};")
    
    # Add title
    tikz_code.append("")
    tikz_code.append("  % Title")
    tikz_code.append("  \\node[above, font=\\small\\bfseries] at (%g,1) {Algorithm Performance Heatmap};" % 
                     (len(algorithms)*cell_width/2))
    
    # Add colorbar
    tikz_code.append("")
    tikz_code.append("  % Colorbar")
    colorbar_x = len(algorithms) * cell_width + 1
    colorbar_height = len(problems) * cell_height * 0.8
    tikz_code.append("  % Colorbar gradient")
    tikz_code.append(f"  \\shade[bottom color=maxcolor, top color=mincolor] ({colorbar_x},-{colorbar_height}) rectangle ({colorbar_x+0.5},0);")
    tikz_code.append(f"  \\draw[black, thin] ({colorbar_x},-{colorbar_height}) rectangle ({colorbar_x+0.5},0);")
    
    # Colorbar labels
    tikz_code.append(f"  \\node[right, font=\\tiny] at ({colorbar_x+0.6},0) {{{min_runtime:.2e}s}};")
    tikz_code.append(f"  \\node[right, font=\\tiny] at ({colorbar_x+0.6},-{colorbar_height}) {{{max_runtime:.2e}s}};")
    tikz_code.append(f"  \\node[rotate=90, above, font=\\tiny] at ({colorbar_x+0.25},-{colorbar_height/2}) {{Runtime (s)}};")
    
    tikz_code.append("\\end{tikzpicture}")
    
    # Also create a standalone LaTeX document
    standalone_code = []
    standalone_code.append("\\documentclass[tikz,border=10pt]{standalone}")
    standalone_code.append("\\usepackage{tikz}")
    standalone_code.append("\\begin{document}")
    standalone_code.extend(tikz_code)
    standalone_code.append("\\end{document}")
    
    return "\n".join(tikz_code), "\n".join(standalone_code)

def main():
    if len(sys.argv) != 2:
        print("Usage: python json_to_tikz.py <json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    try:
        tikz_code, standalone_code = json_to_tikz(json_file)
        
        # Save TikZ code
        tikz_file = json_file.replace('.json', '_tikz.tex')
        with open(tikz_file, 'w') as f:
            f.write(tikz_code)
        print(f"TikZ code saved to: {tikz_file}")
        
        # Save standalone document
        standalone_file = json_file.replace('.json', '_standalone.tex')
        with open(standalone_file, 'w') as f:
            f.write(standalone_code)
        print(f"Standalone LaTeX document saved to: {standalone_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()