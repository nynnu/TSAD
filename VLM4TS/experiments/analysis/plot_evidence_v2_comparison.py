"""Plot score quality separately from routing quality and review cost."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    folder = Path(__file__).parent / 'results/evidence_v2_audited'
    rows = json.loads((folder / 'comparison.json').read_text(encoding='utf-8'))
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    specs = [
        ('Point average precision (threshold-free)',
         [r['point_average_precision']['baseline'] for r in rows],
         [r['point_average_precision']['dynamic_evidence'] for r in rows], 'Average precision'),
        ('Channel recall: SAME tiles and candidate budgets',
         [r['baseline']['event_union_channel_recall'] for r in rows],
         [r['fixed_tile_routing']['dynamic']['event_union_channel_recall'] for r in rows], 'Event-union channel recall'),
        ('GT-normal ticks marked high (lower is better)',
         [r['baseline']['normal_tick_alarm'] for r in rows],
         [r['variant']['normal_tick_alarm'] for r in rows], 'Fraction of GT-normal ticks'),
        ('All-band review pages (cost, not accuracy)',
         [r['baseline']['pages'] for r in rows],
         [r['variant']['pages'] for r in rows], 'Pages before VLM calls'),
    ]
    for ax, (title, old, new, ylabel) in zip(axes.flat, specs):
        ax.bar(x - .19, old, .38, label='Baseline', color='#8494aa')
        ax.bar(x + .19, new, .38, label='Experimental v2', color='#176fa5')
        ax.set_xticks(x, [r['machine'] for r in rows])
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(old + new) * 1.22)
        for i, values in enumerate((old, new)):
            for tick, value in zip(x + (-.19 if i == 0 else .19), values):
                ax.text(tick, value, f'{value:.3f}' if max(old + new) <= 1 else str(value),
                        ha='center', va='bottom', fontsize=9)
    axes[0, 0].legend(frameon=False)
    fig.suptitle('SMD diagnostic cohort: score quality, controlled routing, and cost\n'
                 '3 entities only; no end-to-end VLM performance claim', fontsize=14)
    fig.savefig(folder / 'comparison.png', dpi=160)
    plt.close(fig)
