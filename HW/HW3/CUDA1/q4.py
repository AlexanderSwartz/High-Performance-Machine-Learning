#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('q1_q3.csv')

# X axis is always first column (Million Elements)
x = df.iloc[:, 0]
x_label = 'Million Elements'

def create_plot(columns, output_filename, title=None):
    fig, ax = plt.subplots()
    
    color_map = {'Q1': 'C0'} 
    for col in columns:
        # Group by scenario prefix (e.g. 'Q2 S1')
        if col.startswith('Q1'):
            key = 'Q1'
        else:
            parts = col.split()
            key = ' '.join(parts[:2]) if len(parts) >= 2 else parts[0]
            
        # Assign a new color if we haven't seen this group key yet
        if key not in color_map:
            color_map[key] = f'C{len(color_map)}'

        ls = '--' if 'Warmup' in col else '-'
        # Replace scenario short codes with descriptive legend text
        display_label = col.replace('S1', '1 Block 1 Thread') \
                   .replace('S2', '1 Block 256 Threads') \
                   .replace('S3', 'Multiple Blocks 256 Threads')
        ax.plot(x, df[col], label=display_label, color=color_map[key], linestyle=ls)

    ax.set_xlabel(x_label)
    ax.set_ylabel('Time (s)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_filename)

# Plot 1: q4_without_unified.jpg (Q1 and Q2)
cols_q1_q2 = [c for c in df.columns if c.startswith(('Q1', 'Q2'))]
create_plot(cols_q1_q2, 'q4_without_unified.jpg',
            'Vector Addition on CPU vs Non-Unified GPU')
print("Generating q4_without_unified.jpg")

# Plot 2: q4_with_unified.jpg (Q1 and Q3)
cols_q1_q3 = [c for c in df.columns if c.startswith(('Q1', 'Q3'))]
create_plot(cols_q1_q3, 'q4_with_unified.jpg',
            'Vector Addition on CPU vs Unified GPU')
print("Generating q4_with_unified.jpg")
