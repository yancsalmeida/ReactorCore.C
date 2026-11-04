#!/usr/bin/env python3
import csv
import sys

def generate_svg(csv_path, output_svg):
    times = []
    cas = []
    temps = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time_min']))
            cas.append(float(row['Ca_mol_per_L']))
            temps.append(float(row['temp_K']))

    if not times:
        print("Error: empty CSV file")
        return

    # SVG layout
    w, h = 900, 420
    pad_l, pad_r, pad_t, pad_b = 70, 70, 50, 50
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b

    t_min, t_max = 0.0, max(times)
    ca_min, ca_max = 0.0, 1.05
    t_temp_min, t_temp_max = 345.0, 400.0

    def to_x(t):
        return pad_l + ((t - t_min) / (t_max - t_min)) * plot_w

    def to_y_ca(ca):
        return pad_t + (1.0 - (ca - ca_min) / (ca_max - ca_min)) * plot_h

    def to_y_temp(temp):
        return pad_t + (1.0 - (temp - t_temp_min) / (t_temp_max - t_temp_min)) * plot_h

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" height="100%" style="background:#0b0f19; font-family: monospace;">')

    # Gridlines
    for i in range(7):
        t_val = i * 2.5
        x = to_x(t_val)
        svg.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{pad_t + plot_h}" stroke="#1e293b" stroke-width="1"/>')
        svg.append(f'<text x="{x:.1f}" y="{pad_t + plot_h + 20}" fill="#94a3b8" font-size="11" text-anchor="middle">{t_val:.1f}m</text>')

    # Horizontal gridlines & Y axis labels
    for i in range(5):
        frac = i / 4.0
        y = pad_t + (1.0 - frac) * plot_h
        svg.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" stroke="#1e293b" stroke-width="1"/>')
        
        ca_val = ca_min + frac * (ca_max - ca_min)
        svg.append(f'<text x="{pad_l - 10}" y="{y + 4:.1f}" fill="#38bdf8" font-size="11" text-anchor="end">{ca_val:.2f}</text>')

        temp_val = t_temp_min + frac * (t_temp_max - t_temp_min)
        svg.append(f'<text x="{pad_l + plot_w + 10}" y="{y + 4:.1f}" fill="#f43f5e" font-size="11" text-anchor="start">{temp_val:.0f}K</text>')

    # Event marker at t = 10.0 min
    x_event = to_x(10.0)
    svg.append(f'<line x1="{x_event:.1f}" y1="{pad_t}" x2="{x_event:.1f}" y2="{pad_t + plot_h}" stroke="#f59e0b" stroke-dasharray="4,4" stroke-width="1.5"/>')
    svg.append(f'<text x="{x_event + 6:.1f}" y="{pad_t + 15}" fill="#f59e0b" font-size="11">t=10m: Coolant step increase (300K -> 325K)</text>')

    # Path for Ca (blue)
    ca_pts = [f"{to_x(t):.2f},{to_y_ca(ca):.2f}" for t, ca in zip(times[::5], cas[::5])]
    svg.append(f'<polyline points="{" ".join(ca_pts)}" fill="none" stroke="#38bdf8" stroke-width="2.5"/>')

    # Path for Temp (rose)
    temp_pts = [f"{to_x(t):.2f},{to_y_temp(temp):.2f}" for t, temp in zip(times[::5], temps[::5])]
    svg.append(f'<polyline points="{" ".join(temp_pts)}" fill="none" stroke="#f43f5e" stroke-width="2.5"/>')

    # Titles & Legend
    svg.append(f'<text x="{pad_l}" y="28" fill="#f8fafc" font-size="14" font-weight="bold">ReactorCore.C — CSTR Dynamic Response Trajectory (RK4)</text>')
    svg.append(f'<circle cx="{w - 280}" cy="24" r="5" fill="#38bdf8"/>')
    svg.append(f'<text x="{w - 270}" y="28" fill="#38bdf8" font-size="11">Ca: Concentration (mol/L)</text>')
    svg.append(f'<circle cx="{w - 110}" cy="24" r="5" fill="#f43f5e"/>')
    svg.append(f'<text x="{w - 100}" y="28" fill="#f43f5e" font-size="11">T: Temp (K)</text>')

    svg.append('</svg>')

    with open(output_svg, 'w') as f:
        f.write('\n'.join(svg))
    print(f"[OK] Generated {output_svg}")

if __name__ == '__main__':
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'cstr_simulation.csv'
    out_file = sys.argv[2] if len(sys.argv) > 2 else 'docs/cstr_telemetry.svg'
    generate_svg(csv_file, out_file)
