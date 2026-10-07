"""App. D: the blinded, paired evaluation schematic (`fig:eval-protocol`).

A static version of the project website's protocol animation. Fully schematic: the
shuffles and outcomes are made-up illustrations, not data.

    python -m paper.figures --only overview_eval_protocol
"""

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from mulligan.plotting import paper

POLICIES = [
    ("HG-DAgger", "HG-DAgger", paper.BASELINE),
    ("HG-DAgger+Mulligan", "HG-DAgger\n+Mulligan", paper.OURS),
    ("HiL-IDQL+Mulligan", "HiL-IDQL\n+Mulligan", paper.OURS_RERANK),
]
# perm[i][slot] = policy behind label 'ABC'[slot] at start i; outcome[i][policy].
PERM = [[1, 0, 2], [2, 1, 0], [0, 2, 1], [2, 0, 1], [1, 2, 0], [0, 1, 2]]
OUTCOME = [[0, 1, 1], [1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1]]
DOTS = [[0.2, 0.3], [0.7, 0.8], [0.45, 0.55], [0.85, 0.2], [0.1, 0.75], [0.6, 0.1]]

INK = "#222222"
MUTED = "#666666"
SUCCESS = "#247a4f"
FAILURE = "#d62728"
BLIND_FILL = "#f1f1f1"
BLIND_EDGE = "#9a9a9a"

W, H = paper.fig_size(1.0, height_in=2.35)
ROW0, PITCH = 0.66, 0.235
TOK_H = 0.175


def tint(color, alpha):
    return to_hex([1 - alpha + alpha * c for c in to_rgb(color)])


def row_y(i):
    return ROW0 + i * PITCH


def box(ax, x, y, w, h, *, fc, ec, lw=0.8, r=0.03):
    ax.add_patch(
        FancyBboxPatch(
            (x, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw
        )
    )


def text(ax, x, y, s, **kw):
    kw.setdefault("color", INK)
    kw.setdefault("va", "center")
    ax.text(x, y, s, **kw)


def mark(ax, x, y, ok):
    text(
        ax,
        x,
        y,
        "✓" if ok else "✗",
        color=SUCCESS if ok else FAILURE,
        ha="center",
        fontsize=7.5,
        fontweight="bold",
    )


def token(ax, x, y, w, letter, ok, *, fc, ec):
    box(ax, x, y, w, TOK_H, fc=fc, ec=ec, lw=1.0)
    text(ax, x + 0.1, y, letter, ha="center", fontsize=7, fontweight="bold", color=MUTED)
    mark(ax, x + w - 0.13, y, ok)


def panel_title(ax, x, title, sub):
    text(ax, x, 0.1, title, fontsize=8.5, fontweight="bold")
    text(ax, x, 0.26, sub, fontsize=6.5, color=MUTED)


def build() -> paper.FigureRecord:
    with paper.paper_rc():
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")

        # Left: the arms, hidden behind labels, and the sealed assignment key.
        panel_title(ax, 0.02, "Policies", "hidden from the operator")
        for p, (name, _, color) in enumerate(POLICIES):
            y = row_y(p)
            box(ax, 0.02, y, 1.26, TOK_H, fc="white", ec=color, lw=1.1)
            ax.add_patch(Rectangle((0.09, y - 0.035), 0.07, 0.07, fc=color, ec="none"))
            text(ax, 0.22, y, name, fontsize=6.5)
        y_key = (row_y(3) + row_y(4)) / 2
        box(ax, 0.02, y_key, 1.26, 0.4, fc="#f6f5ef", ec="#d6d2c2")
        text(ax, 0.1, y_key - 0.1, "Assignment key", fontsize=6.5, fontweight="bold")
        text(
            ax,
            0.1,
            y_key + 0.07,
            "sealed until the\nsession ends",
            fontsize=6,
            color=MUTED,
            linespacing=1.1,
        )

        # Middle: the blind session, one placement per row and a fresh shuffle per start.
        x_start, slot_x, tok_w = 1.58, [1.9, 2.31, 2.72], 0.37
        panel_title(ax, 1.42, "Blind session", "the operator sees only A, B, C")
        text(ax, x_start + 0.11, 0.47, "State", ha="center", fontsize=6.5, fontweight="bold")
        text(
            ax,
            (slot_x[0] + slot_x[2] + tok_w) / 2,
            0.47,
            "Rollouts (label, outcome)",
            ha="center",
            fontsize=6.5,
            fontweight="bold",
        )
        for i in range(6):
            y = row_y(i)
            text(ax, x_start - 0.05, y, str(i + 1), ha="right", fontsize=7, color=MUTED)
            box(ax, x_start, y, 0.22, TOK_H, fc="white", ec="#c8c8c8", lw=0.7, r=0.015)
            u, v = DOTS[i]
            ax.plot(x_start + 0.04 + 0.14 * u, y - 0.055 + 0.11 * v, "o", ms=2.6, color=INK)
            for s in range(3):
                p = PERM[i][s]
                token(
                    ax, slot_x[s], y, tok_w, "ABC"[s], OUTCOME[i][p], fc=BLIND_FILL, ec=BLIND_EDGE
                )
        text(ax, x_start, row_y(6) - 0.02, "⋮  44 more states", fontsize=6.5, color=MUTED)

        # Arrow: the key is opened only after the session.
        y_mid = (row_y(2) + row_y(3)) / 2
        ax.add_patch(
            FancyArrowPatch(
                (3.15, y_mid), (3.5, y_mid), arrowstyle="-|>", mutation_scale=9, lw=1.0, color=MUTED
            )
        )
        text(
            ax,
            3.32,
            y_mid - 0.15,
            "open\nkey",
            ha="center",
            fontsize=6.5,
            color=MUTED,
            linespacing=1.0,
        )

        # Right: outcomes regrouped by policy, one row per shared start.
        col_x, col_w = [3.62, 4.24, 4.86], 0.52
        panel_title(ax, 3.55, "After the session", "grouped by policy, paired by initial state")
        for p, (_, short, color) in enumerate(POLICIES):
            text(
                ax,
                col_x[p] + col_w / 2,
                0.47,
                short,
                ha="center",
                fontsize=6.5,
                fontweight="bold",
                color=paper.darken(color, 0.85),
                linespacing=1.0,
            )
        for i in range(6):
            y = row_y(i)
            if len(set(OUTCOME[i])) > 1:
                box(ax, 3.56, y, W - 3.58, PITCH - 0.03, fc="#fbf3d9", ec="none", r=0.02)
            for p, (_, _, color) in enumerate(POLICIES):
                slot = PERM[i].index(p)
                token(
                    ax,
                    col_x[p],
                    y,
                    col_w,
                    "ABC"[slot],
                    OUTCOME[i][p],
                    fc=tint(color, 0.14),
                    ec=color,
                )
        y_tot = row_y(6) - 0.02
        for p, (_, _, color) in enumerate(POLICIES):
            total = sum(row[p] for row in OUTCOME)
            text(
                ax,
                col_x[p] + col_w / 2,
                y_tot,
                f"{total} / 6",
                ha="center",
                fontsize=7.5,
                fontweight="bold",
                color=paper.darken(color, 0.85),
            )
        y_note = H - 0.08
        ax.add_patch(Rectangle((3.56, y_note - 0.04), 0.12, 0.08, fc="#fbf3d9", ec="none"))
        text(ax, 3.72, y_note, "policies differ (paired tests)", fontsize=6.5, color=MUTED)

        return paper.save_paper_figure(fig, "overview_eval_protocol", width_frac=1.0)


if __name__ == "__main__":
    build()
