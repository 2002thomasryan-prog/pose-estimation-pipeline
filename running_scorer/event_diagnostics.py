"""
Dumps Excel tables and PNG plots of gait event detection for inspection.

    from running_scorer.event_diagnostics import export_diagnostics
    export_diagnostics(scorer, output_dir='diagnostics')
"""

import os
import numpy as np

try:
    import openpyxl
    from openpyxl.styles import PatternFill, Font, Alignment
    _HAS_OPENPYXL = True
except ImportError:
    _HAS_OPENPYXL = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


_COLOURS = {
    'td':  '#e74c3c',
    'to':  '#e67e22',
    'ms':  '#2ecc71',
    'mvp': '#3498db',
}

_FILLS = {
    'td':    PatternFill('solid', fgColor='FFE0E0') if _HAS_OPENPYXL else None,
    'to':    PatternFill('solid', fgColor='FFE8CC') if _HAS_OPENPYXL else None,
    'ms':    PatternFill('solid', fgColor='D5F5E3') if _HAS_OPENPYXL else None,
    'mvp':   PatternFill('solid', fgColor='D6EAF8') if _HAS_OPENPYXL else None,
    'p20':   PatternFill('solid', fgColor='FFFDE7') if _HAS_OPENPYXL else None,
    'header': PatternFill('solid', fgColor='2C3E50') if _HAS_OPENPYXL else None,
}


def _extract_signals(kp_arr, kp_map, side, s, e):
    """
    Extract per-frame signals for a single cycle window [s, e).

    Returns dict of 1-D arrays, each length (e - s).
    """
    prefix  = 'r' if side == 'right' else 'l'
    opp     = 'l' if prefix == 'r' else 'r'
    toe_i   = kp_map[f'{prefix}_toe']
    ank_i   = kp_map[f'{prefix}_ankle']
    hip_i   = kp_map[f'{prefix}_hip']
    toe_oi  = kp_map[f'{opp}_toe']

    heel_key = f'{prefix}_heel'
    heel_i   = kp_map.get(heel_key, None)
    if heel_i is not None and heel_i == toe_i:
        heel_i = None

    seg = kp_arr[s:e]
    n   = e - s

    toe_y   = seg[:, toe_i,  1].astype(float)
    ank_x   = seg[:, ank_i,  0].astype(float)
    hip_x   = seg[:, hip_i,  0].astype(float)
    toe_oy  = seg[:, toe_oi, 1].astype(float)

    heel_y  = seg[:, heel_i, 1].astype(float) if heel_i is not None else np.full(n, np.nan)

    ank_hip_xdiff    = np.abs(ank_x - hip_x)
    bilateral_toe_ydiff = np.abs(toe_y - toe_oy)

    def _plateau_flags(y, delta):
        finite = y[np.isfinite(y)]
        if len(finite) < 3:
            return np.zeros(n, bool), np.zeros(n, bool)
        yp = float(np.max(finite)); ym = float(np.min(finite))
        yr = yp - ym if yp > ym else 1.0
        return (np.isfinite(y) & (y >= yp - 0.20 * yr),
                np.isfinite(y) & (y >= yp - 0.09 * yr))

    heel_p20, heel_p9 = _plateau_flags(heel_y, 0.20)
    toe_p20,  toe_p9  = _plateau_flags(toe_y,  0.20)
    plateau_20 = heel_p20 | toe_p20
    plateau_9  = heel_p9  | toe_p9

    return {
        'toe_y':            toe_y,
        'heel_y':           heel_y,
        'ankle_x':          ank_x,
        'hip_x':            hip_x,
        'ankle_hip_xdiff':  ank_hip_xdiff,
        'bilateral_toe_ydiff': bilateral_toe_ydiff,
        'plateau_20':       plateau_20.astype(int),
        'plateau_9':        plateau_9.astype(int),
    }


def _write_excel(scorer, output_path):
    if not _HAS_OPENPYXL:
        print('[event_diagnostics] openpyxl not available — skipping Excel export')
        return

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    summary_rows = []

    for ci, ((s, e), ev) in enumerate(zip(scorer.cycles, scorer.cycle_events)):
        n = e - s
        signals = _extract_signals(scorer.kp_arr, scorer.kp_map, scorer.near_side, s, e)

        td_local  = ev.get('td', 0) or 0
        to_local  = ev.get('to')
        ms_local  = ev.get('ms')
        mvp_local = ev.get('mvp')

        td_g  = s + td_local
        to_g  = (s + to_local)  if to_local  is not None else None
        ms_g  = (s + ms_local)  if ms_local  is not None else None
        mvp_g = (s + mvp_local) if mvp_local is not None else None

        summary_rows.append({
            'cycle':       ci + 1,
            'start_frame': s,
            'end_frame':   e,
            'n_frames':    n,
            'td_global':   td_g,
            'to_global':   to_g   if to_g   is not None else 'N/A',
            'ms_global':   ms_g   if ms_g   is not None else 'N/A',
            'mvp_global':  mvp_g  if mvp_g  is not None else 'N/A',
            'td_local':    td_local,
            'to_local':    to_local if to_local is not None else 'N/A',
            'ms_local':    ms_local if ms_local is not None else 'N/A',
            'mvp_local':   mvp_local if mvp_local is not None else 'N/A',
            'to_local_%':  f'{100*to_local/n:.1f}%'  if to_local  is not None else 'N/A',
            'ms_local_%':  f'{100*ms_local/n:.1f}%'  if ms_local  is not None else 'N/A',
            'mvp_local_%': f'{100*mvp_local/n:.1f}%' if mvp_local is not None else 'N/A',
            'td_signal':        ev.get('td_signal', 'N/A'),
            'to_signal':        ev.get('to_signal', 'N/A'),
            'cycle_len':        ev.get('cycle_len', 'N/A'),
            'cycle_len_outlier': ev.get('cycle_len_outlier', 'N/A'),
            'td_decel_ok':      ev.get('td_decel_ok', 'N/A'),
            'to_accel_ok':      ev.get('to_accel_ok', 'N/A'),
            'mvp_in_flight':    ev.get('mvp_in_flight', 'N/A'),
            'td_plateau_start': ev.get('td_plateau_start', 'N/A'),
            'td_plateau_end':   ev.get('td_plateau_end', 'N/A'),
            'to_plateau_start': ev.get('to_plateau_start', 'N/A'),
            'to_plateau_end':   ev.get('to_plateau_end', 'N/A'),
        })

        ws = wb.create_sheet(title=f'Cycle_{ci+1:02d}')

        headers = [
            'frame_global', 'frame_local',
            'toe_y', 'heel_y', 'ankle_x', 'hip_x',
            'ankle_hip_xdiff', 'bilateral_toe_ydiff',
            'plateau_20', 'plateau_9',
            'event_td', 'event_to', 'event_ms', 'event_mvp',
        ]
        for col, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=h)
            cell.fill = _FILLS['header']
            cell.font = Font(color='FFFFFF', bold=True)
            cell.alignment = Alignment(horizontal='center')

        for local_i in range(n):
            global_i = s + local_i
            row = local_i + 2

            is_td  = (local_i == td_local)
            is_to  = (to_local  is not None and local_i == to_local)
            is_ms  = (ms_local  is not None and local_i == ms_local)
            is_mvp = (mvp_local is not None and local_i == mvp_local)

            fill = None
            if is_td:    fill = _FILLS['td']
            elif is_to:  fill = _FILLS['to']
            elif is_ms:  fill = _FILLS['ms']
            elif is_mvp: fill = _FILLS['mvp']
            elif signals['plateau_20'][local_i]: fill = _FILLS['p20']

            def _val(v):
                return None if (isinstance(v, float) and not np.isfinite(v)) else v

            row_data = [
                global_i, local_i,
                _val(signals['toe_y'][local_i]),
                _val(signals['heel_y'][local_i]),
                _val(signals['ankle_x'][local_i]),
                _val(signals['hip_x'][local_i]),
                _val(signals['ankle_hip_xdiff'][local_i]),
                _val(signals['bilateral_toe_ydiff'][local_i]),
                int(signals['plateau_20'][local_i]),
                int(signals['plateau_9'][local_i]),
                1 if is_td  else 0,
                1 if is_to  else 0,
                1 if is_ms  else 0,
                1 if is_mvp else 0,
            ]

            for col, val in enumerate(row_data, 1):
                cell = ws.cell(row=row, column=col, value=val)
                if fill:
                    cell.fill = fill

        for col in ws.columns:
            max_len = max((len(str(c.value)) for c in col if c.value is not None), default=8)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 20)

    ws_sum = wb.create_sheet(title='Summary', index=0)
    sum_headers = list(summary_rows[0].keys()) if summary_rows else []
    for col, h in enumerate(sum_headers, 1):
        cell = ws_sum.cell(row=1, column=col, value=h)
        cell.fill = _FILLS['header']
        cell.font = Font(color='FFFFFF', bold=True)
        cell.alignment = Alignment(horizontal='center')

    for ri, row_dict in enumerate(summary_rows, 2):
        for col, key in enumerate(sum_headers, 1):
            ws_sum.cell(row=ri, column=col, value=row_dict[key])

    for col in ws_sum.columns:
        max_len = max((len(str(c.value)) for c in col if c.value is not None), default=8)
        ws_sum.column_dimensions[col[0].column_letter].width = min(max_len + 2, 22)

    wb.save(output_path)
    print(f'[event_diagnostics] Excel saved: {output_path}')


def _plot_cycle(ci, s, e, ev, signals, plot_path):
    if not _HAS_MPL:
        return

    n = e - s
    x = np.arange(s, e)

    td_local  = ev.get('td', 0) or 0
    to_local  = ev.get('to')
    ms_local  = ev.get('ms')
    mvp_local = ev.get('mvp')

    td_g  = s + td_local
    to_g  = (s + to_local)  if to_local  is not None else None
    ms_g  = (s + ms_local)  if ms_local  is not None else None
    mvp_g = (s + mvp_local) if mvp_local is not None else None

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'Cycle {ci+1}  (global frames {s}–{e-1},  n={n})',
                 fontsize=13, fontweight='bold')

    def _vlines(ax):
        """Draw vertical event lines on an axis."""
        for g, label, colour in [
            (td_g,  'TD',  _COLOURS['td']),
            (to_g,  'TO',  _COLOURS['to']),
            (ms_g,  'MS',  _COLOURS['ms']),
            (mvp_g, 'MVP', _COLOURS['mvp']),
        ]:
            if g is not None:
                ax.axvline(g, color=colour, linewidth=2, alpha=0.9, zorder=5)
                ax.text(g + 0.1, ax.get_ylim()[1], label,
                        color=colour, fontsize=8, fontweight='bold', va='top')

    ax = axes[0]
    ax.plot(x, signals['heel_y'], color='#2c3e50', linewidth=1.5, label='Near heel Y')
    if not np.all(np.isnan(signals['toe_y'])):
        ax.plot(x, signals['toe_y'], color='#e74c3c', linewidth=1,
                linestyle='--', label='Near toe Y', alpha=0.8)

    p20 = signals['plateau_20'].astype(bool)
    p9  = signals['plateau_9'].astype(bool)
    for i in range(n):
        if p20[i]:
            ax.axvspan(s+i-0.5, s+i+0.5, color='#FFF9C4', alpha=0.5, zorder=0)
        if p9[i]:
            ax.axvspan(s+i-0.5, s+i+0.5, color='#FFE0B2', alpha=0.5, zorder=1)

    ax.set_ylabel('Foot Y (px)\n[higher = lower on screen]')
    ax.legend(fontsize=8, loc='upper right')
    ax.invert_yaxis()
    ax.set_title('Signal 1: Heel Y (black) + Toe Y (red)  —  plateau = TD/TO detection basis', fontsize=9)
    _vlines(ax)

    ax = axes[1]
    ax.plot(x, signals['ankle_hip_xdiff'], color='#27ae60', linewidth=1.5)
    ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
    ax.set_ylabel('|ankle_x - hip_x| (px)')
    ax.set_title('Signal 2: Ankle-Hip X alignment  (minimum = MS)', fontsize=9)
    if ms_g is not None:
        ax.scatter([ms_g], [signals['ankle_hip_xdiff'][ms_local]], 
                   color=_COLOURS['ms'], s=80, zorder=6, label=f'MS={ms_g}')
        ax.legend(fontsize=8)
    _vlines(ax)

    ax = axes[2]
    ax.plot(x, signals['bilateral_toe_ydiff'], color='#2980b9', linewidth=1.5)
    ax.set_ylabel('|near_toe_Y - far_toe_Y| (px)')
    ax.set_title('Signal 3: Bilateral toe Y separation  (minimum = MVP)', fontsize=9)
    if mvp_g is not None:
        ax.scatter([mvp_g], [signals['bilateral_toe_ydiff'][mvp_local]],
                   color=_COLOURS['mvp'], s=80, zorder=6, label=f'MVP={mvp_g}')
        ax.legend(fontsize=8)
    _vlines(ax)

    ax = axes[3]
    ax.plot(x, signals['ankle_x'], color='#e74c3c', linewidth=1.2, label='ankle_x')
    ax.plot(x, signals['hip_x'],   color='#8e44ad', linewidth=1.2, label='hip_x',
            linestyle='--')
    ax.set_ylabel('X position (px)')
    ax.set_xlabel('Global frame index')
    ax.set_title('Signal 4: Ankle X vs Hip X  (cross-over point = MS)', fontsize=9)
    ax.legend(fontsize=8)
    _vlines(ax)

    patches = [mpatches.Patch(color=_COLOURS[k], label=k.upper())
               for k in ['td', 'to', 'ms', 'mvp']]
    fig.legend(handles=patches, loc='lower center', ncol=4,
               fontsize=9, framealpha=0.8,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def _plot_summary(scorer, plot_dir):
    """
    Summary plot: event local positions (as % of cycle) across all cycles.
    Useful to spot outlier cycles immediately.
    """
    if not _HAS_MPL:
        return

    n_cycles = len(scorer.cycles)
    cycle_ids = np.arange(1, n_cycles + 1)

    to_pcts, ms_pcts, mvp_pcts = [], [], []

    for (s, e), ev in zip(scorer.cycles, scorer.cycle_events):
        n = e - s
        to_local  = ev.get('to')
        ms_local  = ev.get('ms')
        mvp_local = ev.get('mvp')
        to_pcts.append(100 * to_local  / n if to_local  is not None else np.nan)
        ms_pcts.append(100 * ms_local  / n if ms_local  is not None else np.nan)
        mvp_pcts.append(100 * mvp_local / n if mvp_local is not None else np.nan)

    fig, ax = plt.subplots(figsize=(max(8, n_cycles * 0.7 + 2), 5))

    ax.scatter(cycle_ids, to_pcts,  color=_COLOURS['to'],  s=70, zorder=5, label='TO %')
    ax.scatter(cycle_ids, ms_pcts,  color=_COLOURS['ms'],  s=70, zorder=5, label='MS %')
    ax.scatter(cycle_ids, mvp_pcts, color=_COLOURS['mvp'], s=70, zorder=5, label='MVP %')

    ax.plot(cycle_ids, to_pcts,  color=_COLOURS['to'],  linewidth=1, alpha=0.5)
    ax.plot(cycle_ids, ms_pcts,  color=_COLOURS['ms'],  linewidth=1, alpha=0.5)
    ax.plot(cycle_ids, mvp_pcts, color=_COLOURS['mvp'], linewidth=1, alpha=0.5)

    ax.axhspan(30, 50, color='#FFE0B2', alpha=0.3, label='Typical TO range (30-50%)')
    ax.axhspan(15, 35, color='#D5F5E3', alpha=0.3, label='Typical MS range (15-35%)')
    ax.axhspan(55, 80, color='#D6EAF8', alpha=0.3, label='Typical MVP range (55-80%)')

    ax.set_xlabel('Cycle number', fontsize=11)
    ax.set_ylabel('Event position (% of cycle length)', fontsize=11)
    ax.set_title('Event Positions Across All Cycles\n'
                 '(TD is always 0% by construction)', fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_xticks(cycle_ids)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plot_dir, 'summary_event_positions.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[event_diagnostics] Summary plot saved: {path}')


def export_diagnostics(scorer, output_dir='diagnostics'):
    """
    Export event detection diagnostics for all cycles.

    Parameters
    ----------
    scorer     : RunningScorer  after .run() has been called
    output_dir : str            directory to write outputs into (created if needed)

    Outputs
    -------
    <output_dir>/events_all_cycles.xlsx
    <output_dir>/plots/cycle_NN_signals.png  (one per cycle)
    <output_dir>/plots/summary_event_positions.png
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    if not scorer.cycles:
        print('[event_diagnostics] No cycles detected — nothing to export.')
        return

    xlsx_path = os.path.join(output_dir, 'events_all_cycles.xlsx')
    _write_excel(scorer, xlsx_path)

    if _HAS_MPL:
        for ci, ((s, e), ev) in enumerate(zip(scorer.cycles, scorer.cycle_events)):
            signals  = _extract_signals(scorer.kp_arr, scorer.kp_map,
                                        scorer.near_side, s, e)
            plot_path = os.path.join(plot_dir, f'cycle_{ci+1:02d}_signals.png')
            _plot_cycle(ci, s, e, ev, signals, plot_path)
            print(f'[event_diagnostics] Plot saved: {plot_path}')

        _plot_summary(scorer, plot_dir)
    else:
        print('[event_diagnostics] matplotlib not available — skipping plots.')

    print(f'[event_diagnostics] Done. Outputs in: {os.path.abspath(output_dir)}')
