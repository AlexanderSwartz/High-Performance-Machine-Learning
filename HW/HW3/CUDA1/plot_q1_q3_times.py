#!/usr/bin/env python3
"""Plot timings from q1-3_times.csv

Creates two PNG files in the same folder:
- times_plot1.png -> columns 2-5 vs column 1
- times_plot2.png -> columns 2 and 6-8 vs column 1

If the CSV has a header, legend labels will use it; otherwise generic labels are used.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Only load from q1_q3.csv per user instruction.
    csv_path = os.path.join(base_dir, 'q1_q3.csv')
    if not os.path.exists(csv_path):
        raise SystemExit('CSV file not found: q1_q3.csv')
    print('Using CSV: q1_q3.csv')

    df = pd.read_csv(csv_path)

    # No table-writing: load only `q1_q3.csv` and proceed to plotting.

    # X axis is first column
    x = df.iloc[:, 0]
    x_label = 'Million Elements'

    # Plot 1: all columns starting with Q1 or Q2
    cols_q1_q2 = [c for c in df.columns if c.startswith('Q1') or c.startswith('Q2')]
    fig1, ax1 = plt.subplots()
    if cols_q1_q2:
        # Group Q2 columns by scenario prefix (e.g. 'Q2 S1') and color each group
        grouped = {}
        for c in cols_q1_q2:
            if c.startswith('Q1'):
                key = 'Q1'
            else:
                parts = c.split()
                key = ' '.join(parts[:2]) if len(parts) >= 2 else parts[0]
            grouped.setdefault(key, []).append(c)

        # Assign colors: Q1 -> C0, then subsequent groups C1, C2, ...
        keys = ['Q1'] + [k for k in grouped.keys() if k != 'Q1']
        color_map = {k: f'C{i}' for i, k in enumerate(keys)}

        for key in keys:
            cols = grouped.get(key, [])
            for col in cols:
                ls = '--' if 'Warm' in col else '-'
                ax1.plot(x, df[col], label=col, color=color_map[key], linestyle=ls)
    else:
        print('No Q1/Q2 columns found for plot 1; skipping')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Time (s)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    fig1.tight_layout()
    out1 = os.path.join(base_dir, 'times_plot1.png')
    fig1.savefig(out1)

    # Plot 2: all columns starting with Q1 or Q3
    cols_q1_q3 = [c for c in df.columns if c.startswith('Q1') or c.startswith('Q3')]
    fig2, ax2 = plt.subplots()
    if cols_q1_q3:
        # Group Q3 columns by scenario prefix (e.g. 'Q3 S1') and color each group
        grouped2 = {}
        for c in cols_q1_q3:
            if c.startswith('Q1'):
                key = 'Q1'
            else:
                parts = c.split()
                key = ' '.join(parts[:2]) if len(parts) >= 2 else parts[0]
            grouped2.setdefault(key, []).append(c)

        # Assign colors: Q1 -> C0, then subsequent groups C1, C2, ...
        keys2 = ['Q1'] + [k for k in grouped2.keys() if k != 'Q1']
        color_map2 = {k: f'C{i}' for i, k in enumerate(keys2)}

        for key in keys2:
            cols = grouped2.get(key, [])
            for col in cols:
                ls = '--' if 'Warm' in col else '-'
                ax2.plot(x, df[col], label=col, color=color_map2[key], linestyle=ls)
    else:
        print('No Q1/Q3 columns found for plot 2; skipping')
    ax2.set_xlabel(x_label)
    ax2.set_ylabel('Time (s)')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    fig2.tight_layout()
    out2 = os.path.join(base_dir, 'times_plot2.png')
    fig2.savefig(out2)

    print(f'Wrote: {out1}\nWrote: {out2}')


if __name__ == '__main__':
    main()
