"""Blog charts: our 2026cur vision-scouting metrics vs Manta (3128 human scouting) and
Statbotics EPA (or TBA component OPR if the Statbotics API is unavailable).
Inputs: data/teams.json (build.py), data/statbotics_team_events_2026cur.json or data/tba_opr_2026cur.json.
Outputs: /home/nick/AI/frc-vision-scorer/docs/blog/img/cmp_*.png (1380 px wide)."""
import json, os, sys, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
OUT = "/home/nick/AI/frc-vision-scorer/docs/blog/img"

# ---- blog look (docs/blog/diagrams/style.css) + dataviz-validated series colours
BG, TEXT, DIM, MUTED = "#0d1220", "#e8ecf5", "#9aa5c0", "#6f7a96"
GRID, AXIS = "#1d2640", "#2b3656"
BLUE, ORANGE, GRAY = "#3d8fe0", "#d95926", "#7c86a2"
plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 9.5, "text.color": TEXT,
    "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": AXIS, "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.labelsize": 9.5,
})
W_IN, DPI = 6.9, 200          # 1380 px
DEBUG = os.environ.get("LABEL_DEBUG")

T = json.load(open(f"{D}/teams.json"))
SBP = f"{D}/statbotics_team_events_2026cur.json"
HAVE_SB = os.path.exists(SBP) and os.path.getsize(SBP) > 1000
M13P = f"{D}/match13_2026cur_matches.json"
HAVE_M13 = os.path.exists(M13P)
if HAVE_SB:
    EXT_NAME, EXT_UNIT = "Statbotics", "EPA"
    ext = {t: {"total": v.get("sb.total"), "auto": v.get("sb.auto"), "teleop": v.get("sb.teleop")}
           for t, v in T.items() if v.get("sb.total") is not None}
elif HAVE_M13:
    # Match13 xP: each team's rating after its LAST qualification match at Curie
    # (playoff matches excluded so the window matches our qual-only metrics).
    EXT_NAME, EXT_UNIT = "Match13", "xP"
    ext = {}
    qms = [m for m in json.load(open(M13P))["matches"] if "_qm" in m["key"]]
    qms.sort(key=lambda m: int(m["key"].split("_qm")[1]))
    for m in qms:
        for t, v in m["teams"].items():
            ext[t] = {"total": v["xpPost"], "auto": v["xAutoPost"], "teleop": v["xTelePost"]}
else:
    EXT_NAME, EXT_UNIT = "TBA OPR", "OPR"
    ext = json.load(open(f"{D}/tba_opr_2026cur.json"))
for t, v in ext.items():
    T.setdefault(t, {}).update({f"ext.{k}": x for k, x in v.items()})


def pts(xk, yk):
    ts = [t for t, v in T.items() if v.get(xk) is not None and v.get(yk) is not None]
    return ts, np.array([T[t][xk] for t in ts], float), np.array([T[t][yk] for t in ts], float)


def rho(xk, yk):
    ts, x, y = pts(xk, yk)
    return spearmanr(x, y)[0], len(ts)


def wrap_lines(s, width):
    out = []
    for para in s.split("\n"):
        out += textwrap.wrap(para, width) or [""]
    return out


def place_labels(ax, fig, xs, ys, labels, all_xy):
    """Greedy placement: 8 directions x 3 radii around each point. Cost = overlap with placed
    labels, other dots, the axes edge, and being nearer another dot than its own. Leader line
    when pushed beyond the first ring."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ab = ax.get_window_extent(r)
    dots = ax.transData.transform(all_xy)
    px = fig.dpi / 72
    placed = []
    dirs = [(1, 0.7, "left", "bottom"), (1, -0.7, "left", "top"), (-1, 0.7, "right", "bottom"),
            (-1, -0.7, "right", "top"), (1.2, 0, "left", "center"), (-1.2, 0, "right", "center"),
            (0, 1.2, "center", "bottom"), (0, -1.2, "center", "top")]
    for x, y, lab in zip(xs, ys, labels):
        own = ax.transData.transform((x, y))
        best = None
        for ring, rad in enumerate((6, 16, 28)):
            for dx, dy, ha, va in dirs:
                t = ax.annotate(lab, (x, y), xytext=(dx * rad, dy * rad), textcoords="offset points",
                                ha=ha, va=va, fontsize=8.5, color=TEXT, fontweight="bold")
                bb = t.get_window_extent(r)
                t.remove()
                pad = bb.expanded(1.06, 1.12)
                cost = sum(pad.overlaps(p) for p in placed) * 20
                cost += sum(1 for d in dots if pad.x0 - 3 * px < d[0] < pad.x1 + 3 * px
                            and pad.y0 - 3 * px < d[1] < pad.y1 + 3 * px) * 3
                if bb.x0 < ab.x0 + 2 or bb.x1 > ab.x1 - 2 or bb.y0 < ab.y0 + 2 or bb.y1 > ab.y1 - 2:
                    cost += 100
                c = np.array([(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2])
                d_own = np.hypot(*(c - own))
                cost += 2 * sum(1 for d in dots if np.hypot(*(c - d)) < d_own - 4 * px)
                cost += ring * 1.5
                # tie-break: keep clear of other dots' neighbourhood (reads as their label)
                for d in dots:
                    if np.hypot(*(d - own)) < 1:
                        continue
                    gx = max(bb.x0 - d[0], 0, d[0] - bb.x1); gy = max(bb.y0 - d[1], 0, d[1] - bb.y1)
                    if np.hypot(gx, gy) < 10 * px:
                        cost += 0.8
                if DEBUG and lab == DEBUG: print(lab, ring, dx, dy, round(cost, 1))
                if best is None or cost < best[0]:
                    best = (cost, dx * rad, dy * rad, ha, va, bb, ring)
        _, dx, dy, ha, va, bb, ring = best
        kw = {}
        if ring > 0:
            kw["arrowprops"] = dict(arrowstyle="-", color=DIM, lw=0.9, shrinkA=2, shrinkB=5, alpha=0.8)
        ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                    fontsize=8.5, color=TEXT, fontweight="bold", zorder=6, **kw)
        placed.append(bb.expanded(1.06, 1.12))


def scatter(fname, xk, yk, title, sub, xlabel, ylabel, label_teams, notes, xfmt=None, yfmt=None,
            xlim=None, ylim=None, n_pairs=None):
    ts, x, y = pts(xk, yk)
    rr, n = spearmanr(x, y)[0], len(ts)
    note_lines = []
    for nt in notes:
        note_lines += wrap_lines(nt, 118) + [None]
    note_lines = note_lines[:-1] if note_lines else []
    head_in, ax_h, xl_in = 0.86, 3.55, 0.48
    notes_in = 0.19 * len(note_lines) + (0.14 if note_lines else 0)
    H = head_in + ax_h + xl_in + notes_in + 0.12
    fig = plt.figure(figsize=(W_IN, H), dpi=DPI)
    L, R = 0.70, 0.22
    ax = fig.add_axes([L / W_IN, (xl_in + notes_in + 0.12) / H, (W_IN - L - R) / W_IN, ax_h / H])
    fig.text(0.26 / W_IN, 1 - 0.30 / H, title, fontsize=13.5, fontweight="bold", color=TEXT, va="baseline")
    stat = f"Spearman ρ = {rr:+.2f}  ·  n = {n} teams" + (f" ({n_pairs} team-matches)" if n_pairs else "")
    fig.text(0.26 / W_IN, 1 - 0.55 / H, stat, fontsize=10, color=TEXT, va="baseline", fontweight="bold")
    fig.text(0.26 / W_IN, 1 - 0.75 / H, sub, fontsize=9, color=DIM, va="baseline")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=4)
    groups = [tuple(str(u) for u in (g if isinstance(g, tuple) else (g,))) for g in label_teams]
    lab_set = {u for g in groups for u in g}
    m = np.array([t in lab_set for t in ts])
    ax.scatter(x[~m], y[~m], s=30, color=BLUE, alpha=0.55, lw=0, zorder=3)
    ax.scatter(x[m], y[m], s=42, color=BLUE, edgecolor=BG, lw=1.2, zorder=4)
    # light trend line (least squares, for the eye only; rho is rank-based)
    k, b = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 50)
    ax.plot(xx, k * xx + b, color=DIM, lw=1.1, alpha=0.55, zorder=2, solid_capstyle="round")
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    if xfmt: ax.xaxis.set_major_formatter(xfmt)
    if yfmt: ax.yaxis.set_major_formatter(yfmt)
    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)
    groups = [g for g in groups if all(u in ts for u in g)]
    gx = np.array([np.mean([x[ts.index(u)] for u in g]) for g in groups])
    gy = np.array([np.mean([y[ts.index(u)] for u in g]) for g in groups])
    place_labels(ax, fig, gx, gy, [" · ".join(g) for g in groups], np.c_[x, y])
    yy = (notes_in + 0.06) / H
    for ln in note_lines:
        if ln is None:
            yy -= 0.07 / H
            continue
        fig.text(0.26 / W_IN, yy, ln, fontsize=8.6, color=DIM, va="top")
        yy -= 0.19 / H
    fig.savefig(f"{OUT}/{fname}", dpi=DPI)
    plt.close(fig)
    print(f"{fname}: rho {rr:+.2f} n {n}")
    return rr, n


pct = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}")
sec = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f} s")
NP = 702

if "scatter" in sys.argv or len(sys.argv) == 1:
    scatter("cmp_defense_manta.png", "joined.opp_frac|attacking", "manta.attacking",
            "Defense: time in the opponent's zone vs. scouted defense",
            "Each dot is a team, averaged over the qualification matches both we and Manta covered.",
            "Ours: share of tracked teleop time in or within 1 m of the opponent's alliance zone",
            "Manta: defense played (s per match)",
            [4786, 2682, 571, 7028, 294, 5907],
            ["Biggest miss, 4786: scouts log 48 s of defense a match (3rd most), but we put it in the opponent's "
             "zone only 4% of the time. Their notes say why: \"bump/trench defending\", \"opportunistic defense in nz\", "
             "\"trapped 4160 on their side\". It defends in the neutral zone and at the bumps, which our "
             "zone-based score does not count. 571 and 7028 are the same pattern (\"def in NZ\", \"guarded trench\").",
             "We over-rank 294 and 5907: they spend inactive periods in the opponent's zone stealing fuel (scouts: \"steal feeds OAZ\", "
             "\"go to opposing az and steal\"), which looks the same as defending to a position-only metric."],
            xfmt=pct, yfmt=sec, n_pairs=NP)

    scatter("cmp_cycles_manta.png", "joined.cyc_per_min|scoring", "manta.scoring",
            "Teleop cycles vs. scouted fuel scored",
            "Each dot is a team, averaged over the qualification matches both we and Manta covered.",
            "Ours: teleop cycles (neutral zone → own alliance zone) per tracked minute",
            "Manta: teleop fuel scored per match",
            [7567, 254, 125, 5199],
            ["We count trips, scouts count fuel. 254 and 125 are two of the top three fuel scorers but sit mid-pack "
             "on cycles: scouts note they shoot \"from anywhere\", even from the far side of the neutral zone, "
             "so they rarely need to drive back into their own zone, and each trip carries a full hopper.",
             "7567 is the opposite: our 3rd-most cycles, but scouts note trouble intaking (\"cycles were very "
             "slow\"), so many trips moved little fuel. 5199 was tracked only "
             "41% of teleop; tracking gaps reset the cycle counter, so it is undercounted."],
            yfmt=None, n_pairs=NP)

    scatter("cmp_speed_manta.png", "joined.avg_speed|speed", "manta.speed",
            "Average speed vs. what scouts call fast",
            "Manta speed = mean of scout note words per match: fast = +1, average = 0, slow = −1.",
            "Ours: average speed over all tracked time (m/s), stops included",
            "Manta: scouts' speed words (mean)",
            [6615, (571, 5712), 3128],
            ["Scouts rate the drivetrain; we measure how much the robot actually drives, stops included. 6615 gets "
             "\"fast dt\" in most notes but is stopped half of its tracked time (event median 38%), usually parked "
             "3–5 ft from the hub shooting. 571 (stopped 44%) and 5712 (1.44 m/s while moving, median 1.52) "
             "are also called fast but rank low on average speed.",
             "The scouts' scale is squeezed at the top: most teams are called \"fast\" in almost every note, so "
             "+0.8 vs +1.0 is one or two words. 3128 (our fastest by average speed) scouted itself as average-to-fast."],
            xfmt=matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}"), n_pairs=NP)

    scatter("cmp_bump_manta.png", "joined.stuck_s|beached", "manta.beached",
            "Stopped on a bump vs. scouts writing \"beached\"",
            "39 of 702 scouted robot-matches mention beaching; most teams never get the note.",
            "Ours: seconds per match stopped (< 0.5 m/s) with the robot's centre on a bump",
            "Manta: share of matches noted \"beached\"",
            [11167, (2682, 3932, 6510), 4027, 5712, 589, 9200, 9068],
            ["Where it lines up: 4027, 5712 and 589 are near the top of both. Where we miss, the robot usually "
             "drops out of tracking on or next to the bump: 11167 (noted beached in 4 of 10 matches) has a 15 s "
             "tracking gap starting on a bump in qm62 and 6–9 s gaps starting within 1.1 m of one in qm9; 6510 "
             "qm86 has an 18.8 s gap starting 1 m from a bump. 2682 qm34 (\"beached for 28\") is tracked stopped "
             "for 67 s, but 1–1.6 m from any bump outline. 3932 is a note-matching false positive: \"kept 7421 "
             "stuck on the bump\" is about the robot it was defending.",
             "Where scouts are silent: 9200 sat on a bump 14 s in qm86 — scouts logged it 95 s out of commission "
             "(\"oofed in front of hub\"), not beached. 9068 stopped on a bump 19 s in qm9 with no note."],
            xfmt=matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f} s"), yfmt=pct, n_pairs=NP)

    if HAVE_M13 and not HAVE_SB:
        ext_args = ("cmp_cycles_match13.png", "Teleop cycles vs. Match13 teleop rating",
                    "Match13 xTele (teleop points per robot) after each team's last Curie qual; ours over all counted quals.",
                    "Match13: teleop xP (points)", [7567, 5199, 254, 125, 10254, 3792, 8613],
                    "Same trips-vs-fuel gap as against the scouts. 254, 125 and 10254 score from far out with big hoppers "
                    "(scouts: \"shoot from anywhere\", \"big hopper\"), so few trips; 7567 and 3792 make many trips that move little "
                    "fuel (3792 often defended or fed its human player). 8613 is a Match13 disagreement too: scouts credit it "
                    "with ~143 fuel a match (median 113), Match13 with a below-median teleop rating. 5199 is undercounted "
                    "(41% of teleop tracked). ")
    elif HAVE_SB:
        ext_args = ("cmp_cycles_statbotics.png", "Teleop cycles vs. Statbotics teleop EPA",
                    "Statbotics teleop points per robot; ours over all counted qualification matches.",
                    "Statbotics: teleop EPA (points)", [7567, 5199, 10254, 125, 254, 3792],
                    "Same trips-vs-fuel gap as against the scouts. ")
    else:
        ext_args = ("cmp_cycles_opr.png", "Teleop cycles vs. teleop OPR (from TBA scores)",
                    "Least-squares OPR from TBA teleop points per robot; ours over all counted qualification matches.",
                    "Teleop OPR (points per robot)", [7567, 5199, 166, 10254, 125, 254, 9609, 3792],
                    "Same trips-vs-fuel gap as against the scouts. 254, 125 and 10254 score from far out with big hoppers "
                    "(scouts: \"shoot from anywhere\", \"big hopper\"), so few trips; 7567 and 3792 make many trips that move little "
                    "fuel (3792 often defended or fed its human player). 166 and 9609 are an OPR disagreement too: scouts "
                    "credit them with ~150 fuel a match (median 113), OPR with below-median points. 5199 is undercounted "
                    "(41% of teleop tracked). ")
    fname, title, sub, ylab, lab_teams, note = ext_args
    scatter(fname, "cycles.cycles_per_min", "ext.teleop", title, sub,
            "Ours: teleop cycles per tracked minute (app ranking score)", ylab, lab_teams,
            [note + f"For reference, Manta's scouted teleop fuel matches {EXT_NAME} teleop at "
             f"ρ = {rho('manta.scoring', 'ext.teleop')[0]:+.2f}: counting fuel beats counting trips, and we do not see fuel."],
            xfmt=matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}"))


# ---- Match13 small multiples
def grid_match13():
    if not (HAVE_M13 and not HAVE_SB):
        return
    panels = [
        ("Teleop cycles vs. teleop rating", "cycles.cycles_per_min", "ext.teleop",
         "Teleop cycles per tracked minute", "Match13 teleop xP", [254, 7567, 3792],
         lambda v, _: f"{v:.1f}"),
        ("Auto cycles vs. auto rating", "cycles.cycles_auto_per_match", "ext.auto",
         "Auto cycles per match", "Match13 auto xP", [3932, 6510],
         lambda v, _: f"{v:.1f}"),
        ("Average speed vs. overall rating", "movement.avg_speed_mps", "ext.total",
         "Average speed while tracked (m/s)", "Match13 xP", [4786, 3792, 294, 10254],
         lambda v, _: f"{v:.1f}"),
        ("Defense vs. overall rating", "defense.opp_side_frac", "ext.total",
         "Share of teleop on the opponent's side", "Match13 xP", [2682, 4786, 5907],
         lambda v, _: f"{v:.0%}"),
    ]
    notes = [
        "Each dot is a team: our metric (x) from video positions only, against Match13's rating (y) after the team's last "
        "Curie qualification match. We do not count fuel; these compare where robots drive with how much they score.",
        "Teleop: 254 shoots from far out, so few trips; 7567 and 3792 make many trips that move little fuel. "
        "Auto: 3932's routines score from near its own wall without a trip to the neutral zone, so 0 auto cycles; "
        "6510 makes trips that Match13 does not credit with many points. Speed: 4786 and 3792 drive a lot for their rating (both play a lot "
        "of defense); 294 and 10254 score a lot without moving much. Defense runs the other way (negative): "
        "the robots on the opponent's side most are mostly lower scorers, led by 2682. 4786 defends in the neutral "
        "zone, which this metric misses; 5907 goes over to steal fuel.",
    ]
    note_lines = []
    for nt in notes:
        note_lines += wrap_lines(nt, 118) + [None]
    note_lines = note_lines[:-1]
    head_in, pan_h, pan_title, xl_in, gap_in = 0.78, 2.15, 0.46, 0.42, 0.14
    notes_in = 0.19 * len(note_lines) + 0.14
    H = head_in + 2 * (pan_title + pan_h + xl_in) + gap_in + notes_in + 0.1
    fig = plt.figure(figsize=(W_IN, H), dpi=DPI)
    fig.text(0.26 / W_IN, 1 - 0.32 / H, "Our metrics vs. Match13 team ratings", fontsize=13.5,
             fontweight="bold", color=TEXT, va="baseline")
    fig.text(0.26 / W_IN, 1 - 0.56 / H, "Curie 2026 quals, 74 teams  ·  ρ = rank correlation (1 = same order, negative = opposite order)", fontsize=9, color=DIM, va="baseline")
    L, mid, R = 0.62, 0.62, 0.16
    pw = (W_IN - L - mid - R) / 2
    for i, (ttl, xk, yk, xl, yl, labs, xf) in enumerate(panels):
        r_, c_ = divmod(i, 2)
        top = head_in + r_ * (pan_title + pan_h + xl_in + gap_in) + pan_title
        x0 = L + c_ * (pw + mid)
        ax = fig.add_axes([x0 / W_IN, 1 - (top + pan_h) / H, pw / W_IN, pan_h / H])
        ts, x, y = pts(xk, yk)
        rr = spearmanr(x, y)[0]
        fig.text((x0 - 0.36) / W_IN, 1 - (top - 0.12) / H, ttl, fontsize=9.8, fontweight="bold", color=TEXT, va="baseline")
        fig.text((x0 + pw) / W_IN, 1 - (top - 0.12) / H, f"ρ = {rr:+.2f}", fontsize=9.8, fontweight="bold",
                 color=ORANGE if rr >= 0 else GRAY, va="baseline", ha="right")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color(AXIS)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=3, labelsize=8)
        lab_set = {str(t) for t in labs}
        m = np.array([t in lab_set for t in ts])
        ax.scatter(x[~m], y[~m], s=16, color=BLUE, alpha=0.55, lw=0, zorder=3)
        ax.scatter(x[m], y[m], s=24, color=BLUE, edgecolor=BG, lw=1.0, zorder=4)
        k, b = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 50)
        ax.plot(xx, k * xx + b, color=DIM, lw=1.0, alpha=0.55, zorder=2)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xf))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.set_xlabel(xl, labelpad=4, fontsize=8.6)
        ax.set_ylabel(yl, labelpad=4, fontsize=8.6)
        lt = [str(t) for t in labs if str(t) in ts]
        place_labels(ax, fig, np.array([x[ts.index(t)] for t in lt]), np.array([y[ts.index(t)] for t in lt]),
                     lt, np.c_[x, y])
        print(f"grid panel {ttl}: rho {rr:+.2f} n {len(ts)}")
    yy = (notes_in + 0.02) / H
    for ln in note_lines:
        if ln is None:
            yy -= 0.07 / H
            continue
        fig.text(0.26 / W_IN, yy, ln, fontsize=8.6, color=DIM, va="top")
        yy -= 0.19 / H
    fig.savefig(f"{OUT}/cmp_match13_grid.png", dpi=DPI)
    plt.close(fig)


# ---- summary bars
def summary():
    rows = [  # label, xk, yk, sign, kind
        ("Auto cycles vs. scouted auto fuel", "joined.cyc_auto|auto_scoring", "manta.auto_scoring", 1, "manta"),
        ("Opponent-zone time vs. scouted defense", "joined.opp_frac|attacking", "manta.attacking", 1, "manta"),
        ("Teleop cycles/min vs. scouted teleop fuel", "joined.cyc_per_min|scoring", "manta.scoring", 1, "manta"),
        ("Average speed vs. scouts' speed words", "joined.avg_speed|speed", "manta.speed", 1, "manta"),
        ("Share of time moving vs. scouted dead time*", "joined.moving_frac|oof", "manta.oof", -1, "manta"),
        ("Stopped on a bump vs. \"beached\" notes", "joined.stuck_s|beached", "manta.beached", 1, "manta"),
        ("Teleop cycles/min vs. scouted feeding", "joined.cyc_per_min|feeding", "manta.feeding", 1, "manta"),
        (f"Teleop cycles/min vs. teleop {EXT_UNIT}", "cycles.cycles_per_min", "ext.teleop", 1, "ext"),
        (f"Auto cycles vs. auto {EXT_UNIT}", "cycles.cycles_auto_per_match", "ext.auto", 1, "ext"),
        (f"Average speed vs. total {EXT_UNIT}", "movement.avg_speed_mps", "ext.total", 1, "ext"),
        (f"Reference: scouted teleop fuel vs. teleop {EXT_UNIT}", "manta.scoring", "ext.teleop", 1, "ref"),
    ]
    vals = []
    for lab, xk, yk, s, kind in rows:
        r, n = rho(xk, yk)
        vals.append((lab, s * r, n, kind))
    ref = [v for v in vals if v[3] == "ref"]
    vals = sorted([v for v in vals if v[3] != "ref"], key=lambda v: -v[1]) + ref
    col = {"manta": BLUE, "ext": ORANGE, "ref": GRAY}
    n_rows = len(vals)
    row_h = 0.34
    head_in, foot_in = 1.12, 1.05
    ax_h = row_h * n_rows
    H = head_in + ax_h + foot_in
    fig = plt.figure(figsize=(W_IN, H), dpi=DPI)
    L_in, R_in = 3.05, 0.45
    ax = fig.add_axes([L_in / W_IN, foot_in / H, (W_IN - L_in - R_in) / W_IN, ax_h / H])
    fig.text(0.26 / W_IN, 1 - 0.32 / H, "What lines up, and what doesn't", fontsize=13.5, fontweight="bold", va="baseline")
    fig.text(0.26 / W_IN, 1 - 0.56 / H,
             f"Spearman rank correlation per team, 2026 Curie qualifications ({min(v[2] for v in vals)}–{max(v[2] for v in vals)} teams).",
             fontsize=9, color=DIM, va="baseline")
    # legend row
    lx = 0.26
    for name, c in (("vs. Manta human scouting", BLUE), (f"vs. {EXT_NAME}", ORANGE), ("Reference (not ours)", GRAY)):
        fig.patches.append(matplotlib.patches.FancyBboxPatch(
            (lx / W_IN, 1 - 0.93 / H), 0.12 / W_IN, 0.12 / H, boxstyle="round,pad=0,rounding_size=0.004",
            transform=fig.transFigure, color=c))
        t = fig.text((lx + 0.19) / W_IN, 1 - 0.875 / H, name, fontsize=9, color=DIM, va="center")
        fig.canvas.draw()
        lx += 0.19 + t.get_window_extent().width / DPI + 0.3
    y = np.arange(n_rows)[::-1]
    ax.set_xlim(0, 1.0)
    ax.set_ylim(min(y) - 0.85, max(y) + 0.45)
    # data-y per data-x at equal screen length, so the round corners are circular
    ASPECT = ((max(y) + 0.45 - min(y) + 0.85) / ax_h) / (1.0 / (W_IN - L_in - R_in))
    for i, (lab, r, n, kind) in enumerate(vals):
        yi = y[i] - (0.45 if kind == "ref" else 0)
        bh = 0.56
        ax.add_patch(matplotlib.patches.FancyBboxPatch((0, yi - bh / 2), r, bh, boxstyle="round,pad=0,rounding_size=0.012",
                     mutation_aspect=ASPECT, color=col[kind], lw=0, zorder=3))
        ax.add_patch(matplotlib.patches.Rectangle((0, yi - bh / 2), min(0.02, r), bh, color=col[kind], lw=0, zorder=3))
        ax.text(r + 0.02 if r >= 0 else r - 0.02, yi, f"{r:+.2f}", va="center", ha="left" if r >= 0 else "right",
                fontsize=9, color=TEXT, fontweight="bold")
    ax.set_yticks([y[i] - (0.45 if v[3] == "ref" else 0) for i, v in enumerate(vals)])
    ax.set_yticklabels([v[0] for v in vals], fontsize=9, color=TEXT)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(min(y) - 0.85, max(y) + 0.45)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.25))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".") if v else "0"))
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)
    ref_y = y[-1] - 0.45 + 0.72
    ax.plot([-(L_in - 0.26) / (W_IN - L_in - R_in), 1.0], [ref_y, ref_y], color=AXIS, lw=0.8, clip_on=False, transform=ax.get_yaxis_transform())
    notes = ("Manta rows use only the 702 team-matches both sources cover. 1 = identical team ranking, 0 = unrelated. "
             "*Sign flipped: more time moving should mean less dead time. We do not measure feeding (a feeder "
             "pushes fuel to a partner without a return trip), so its low score is expected.")
    yy = (foot_in - 0.44) / H
    for ln in wrap_lines(notes, 120):
        fig.text(0.26 / W_IN, yy, ln, fontsize=8.6, color=DIM, va="top")
        yy -= 0.19 / H
    fname = "cmp_summary.png"
    fig.savefig(f"{OUT}/{fname}", dpi=DPI)
    plt.close(fig)
    for v in vals:
        print(f"  {v[1]:+.2f}  n {v[2]}  {v[3]:5s} {v[0]}")


if "summary_old" in sys.argv:
    summary()

if "grid" in sys.argv:
    grid_match13()


# ---- paired scatters: our metric vs Manta (left) and vs Match13 (right)
def _panel(fig, x0, top, pw, ph, spec, H):
    ttl, xk, yk, xl, yl, labs, xf, yf = spec
    ax = fig.add_axes([x0 / W_IN, 1 - (top + ph) / H, pw / W_IN, ph / H])
    ts, x, y = pts(xk, yk)
    rr = spearmanr(x, y)[0]
    fig.text((x0 - 0.40) / W_IN, 1 - (top - 0.12) / H, ttl, fontsize=9.8, fontweight="bold", color=TEXT, va="baseline")
    fig.text((x0 + pw) / W_IN, 1 - (top - 0.12) / H, f"ρ = {rr:+.2f}", fontsize=9.8, fontweight="bold",
             color=TEXT if rr >= 0 else GRAY, va="baseline", ha="right")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(AXIS)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=3, labelsize=8)
    groups = [tuple(str(u) for u in (g if isinstance(g, tuple) else (g,))) for g in labs]
    lab_set = {u for g in groups for u in g}
    m = np.array([t in lab_set for t in ts])
    col = BLUE if "manta" in yk else ORANGE
    ax.scatter(x[~m], y[~m], s=16, color=col, alpha=0.55, lw=0, zorder=3)
    ax.scatter(x[m], y[m], s=24, color=col, edgecolor=BG, lw=1.0, zorder=4)
    k, b = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 50)
    ax.plot(xx, k * xx + b, color=DIM, lw=1.0, alpha=0.55, zorder=2)
    if xf: ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xf))
    if yf: ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(yf))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.set_xlabel(xl, labelpad=4, fontsize=8.6)
    ax.set_ylabel(yl, labelpad=4, fontsize=8.6)
    groups = [g for g in groups if all(u in ts for u in g)]
    gx = np.array([np.mean([x[ts.index(u)] for u in g]) for g in groups])
    gy = np.array([np.mean([y[ts.index(u)] for u in g]) for g in groups])
    place_labels(ax, fig, gx, gy, [" · ".join(g) for g in groups], np.c_[x, y])
    return rr, len(ts)


def pair(fname, title, sub, left, right, notes):
    note_lines = []
    for nt in notes:
        note_lines += wrap_lines(nt, 118) + [None]
    note_lines = note_lines[:-1]
    head_in, pan_title, ph, xl_in = 0.78, 0.44, 2.45, 0.42
    notes_in = 0.19 * len(note_lines) + 0.14
    H = head_in + pan_title + ph + xl_in + notes_in + 0.1
    fig = plt.figure(figsize=(W_IN, H), dpi=DPI)
    fig.text(0.26 / W_IN, 1 - 0.32 / H, title, fontsize=13.5, fontweight="bold", color=TEXT, va="baseline")
    fig.text(0.26 / W_IN, 1 - 0.56 / H, sub, fontsize=9, color=DIM, va="baseline")
    L, mid, R = 0.66, 0.66, 0.16
    pw = (W_IN - L - mid - R) / 2
    top = head_in + pan_title
    r1 = _panel(fig, L, top, pw, ph, left, H)
    r2 = _panel(fig, L + pw + mid, top, pw, ph, right, H)
    yy = (notes_in + 0.02) / H
    for ln in note_lines:
        if ln is None:
            yy -= 0.07 / H
            continue
        fig.text(0.26 / W_IN, yy, ln, fontsize=8.6, color=DIM, va="top")
        yy -= 0.19 / H
    fig.savefig(f"{OUT}/{fname}", dpi=DPI)
    plt.close(fig)
    print(f"{fname}: Manta {r1[0]:+.2f} (n {r1[1]})  Match13 {r2[0]:+.2f} (n {r2[1]})")


F1 = lambda v, _: f"{v:.1f}"
PCT = lambda v, _: f"{v:.0%}"
SEC = lambda v, _: f"{v:.0f} s"
SUB = "Each dot is a team, 2026 Curie quals. Left: the 702 team-matches Manta also scouted. Right: all quals."


def pairs():
    pair("cmp_defense.png", "Defense: time on the opponent's side", SUB,
         ("vs. Manta scouted defense", "joined.opp_frac|attacking", "manta.attacking",
          "Ours: share of teleop in the opponent's zone", "Manta: defense played (s/match)", [2682, 4786, 5907],
          PCT, SEC),
         ("vs. Match13 overall rating", "defense.opp_side_frac", "ext.total",
          "Ours: share of teleop in the opponent's zone", "Match13 xP", [2682, 4786, 5907], PCT, None),
         ["Scouts agree on who plays defense: 2682 is #1 for both. The biggest miss is 4786: scouts log 48 s of defense "
          "a match, but it defends in the neutral zone and at the bumps (\"bump/trench defending\", \"opportunistic defense in nz\"), which "
          "a metric counting time in the opponent's zone can't see. 5907 goes over to steal fuel, which looks the same as defending.",
          "Match13 has no defense rating, so the right panel is against overall rating, and it runs the other way: "
          "the robots that spend the most time on the opponent's side are mostly the lower scorers."])
    pair("cmp_cycles.png", "Teleop cycles: trips home from the neutral zone", SUB,
         ("vs. Manta scouted teleop fuel", "joined.cyc_per_min|scoring", "manta.scoring",
          "Ours: teleop cycles per tracked minute", "Manta: teleop fuel scored / match", [254, 125, 7567, 5199],
          F1, None),
         ("vs. Match13 teleop rating", "cycles.cycles_per_min", "ext.teleop",
          "Ours: teleop cycles per tracked minute", "Match13 teleop xP", [254, 7567, 3792, 8613], F1, None),
         ["We count trips, not fuel. 254 and 125 are top fuel scorers but mid-pack on cycles: they shoot from anywhere, "
          "so they rarely drive home. 7567 makes the 3rd-most trips but scouts note slow intake; 3792 often defended or fed "
          "its human player. 5199 was tracked only 41% of teleop, and tracking gaps break up cycles. 8613 is a Match13 "
          "disagreement: scouts credit it with ~143 fuel a match (median 113), Match13 with a below-median teleop rating."])
    pair("cmp_auto.png", "Auto cycles: trips home from the neutral zone in auto", SUB,
         ("vs. Manta scouted auto fuel", "joined.cyc_auto|auto_scoring", "manta.auto_scoring",
          "Ours: auto cycles per match", "Manta: auto fuel scored / match", [(1987, 7028), 9023], F1, None),
         ("vs. Match13 auto rating", "cycles.cycles_auto_per_match", "ext.auto",
          "Ours: auto cycles per match", "Match13 auto xP", [(1987, 7028), 6510], F1, None),
         ["The top of the chart agrees: 1987 and 7028 make the most auto trips (1.7-1.8 a match) and are top 8 in both "
          "scouted auto fuel and Match13's auto rating; 9 of our top 10 are in the top quarter of both. Misses: 9023 drives "
          "7-9 m out but never gets back before auto ends in the matches Manta scouted, so no completed cycle. 6510 makes plenty of trips that Match13 rates low "
          "(scouts give it more auto fuel than Match13 does)."])
    pair("cmp_speed.png", "Average speed", SUB,
         ("vs. Manta scouts' speed words", "joined.avg_speed|speed", "manta.speed",
          "Ours: average speed incl. stops (m/s)", "Manta: fast +1 / avg 0 / slow -1", [6615, 3128], F1, None),
         ("vs. Match13 overall rating", "movement.avg_speed_mps", "ext.total",
          "Ours: average speed incl. stops (m/s)", "Match13 xP", [4786, 3792, 294, 10254], F1, None),
         ["Scouts rate the drivetrain; we measure how much the robot actually drives. 6615 is called \"fast dt\" but is "
          "stopped half its tracked time, parked shooting. Scouts call most teams fast, so their scale is squeezed at the top; "
          "3128, our fastest, scouted itself as average-to-fast. "
          "Against Match13, 4786 and 3792 drive a lot for their rating (both play a lot of defense), while 294 and 10254 "
          "score a lot without moving much."])


def summary_grouped():
    rows = [  # metric, manta label, (xk, yk, sign), match13 label, (xk, yk) or None
        ("Auto cycles", "scouted auto fuel", ("joined.cyc_auto|auto_scoring", "manta.auto_scoring", 1),
         "auto xP", ("cycles.cycles_auto_per_match", "ext.auto")),
        ("Opponent-zone time", "scouted defense", ("joined.opp_frac|attacking", "manta.attacking", 1), None, None),
        ("Teleop cycles / min", "scouted teleop fuel", ("joined.cyc_per_min|scoring", "manta.scoring", 1),
         "teleop xP", ("cycles.cycles_per_min", "ext.teleop")),
        ("Average speed", "scouts' speed words", ("joined.avg_speed|speed", "manta.speed", 1),
         "overall xP", ("movement.avg_speed_mps", "ext.total")),
        ("Share of time moving", "scouted dead time*", ("joined.moving_frac|oof", "manta.oof", -1), None, None),
        ("Stopped on a bump", "\"beached\" notes", ("joined.stuck_s|beached", "manta.beached", 1), None, None),
    ]
    vals = []
    for metric, ml, (mx, my, sg), xl, xp in rows:
        rm = sg * rho(mx, my)[0]
        rx = rho(*xp)[0] if xp else None
        vals.append((metric, ml, rm, xl, rx))
    vals.sort(key=lambda v: -v[2])
    ref = rho("manta.scoring", "ext.teleop")[0]
    n_rows = len(vals)
    row_h = 0.62
    head_in, foot_in = 1.12, 1.25
    ax_h = row_h * n_rows + 0.55
    H = head_in + ax_h + foot_in
    fig = plt.figure(figsize=(W_IN, H), dpi=DPI)
    L_in, R_in = 2.45, 0.45
    ax = fig.add_axes([L_in / W_IN, foot_in / H, (W_IN - L_in - R_in) / W_IN, ax_h / H])
    fig.text(0.26 / W_IN, 1 - 0.32 / H, "What lines up, and what doesn't", fontsize=13.5, fontweight="bold", va="baseline")
    fig.text(0.26 / W_IN, 1 - 0.56 / H, "Spearman rank correlation per team, 2026 Curie qualifications (72-74 teams).",
             fontsize=9, color=DIM, va="baseline")
    lx = 0.26
    for name, c in (("vs. Manta human scouting", BLUE), ("vs. Match13 rating", ORANGE), ("Reference (not ours)", GRAY)):
        fig.patches.append(matplotlib.patches.FancyBboxPatch(
            (lx / W_IN, 1 - 0.93 / H), 0.12 / W_IN, 0.12 / H, boxstyle="round,pad=0,rounding_size=0.004",
            transform=fig.transFigure, color=c))
        t = fig.text((lx + 0.19) / W_IN, 1 - 0.875 / H, name, fontsize=9, color=DIM, va="center")
        fig.canvas.draw()
        lx += 0.19 + t.get_window_extent().width / DPI + 0.3
    ymax = n_rows * 1.0
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-1.15, ymax)
    bh = 0.30
    def bar(yc, r, c):
        ax.add_patch(matplotlib.patches.Rectangle((0, yc - bh / 2), r, bh, color=c, lw=0, zorder=3))
        ax.text(r + 0.015, yc, f"{r:+.2f}", va="center", ha="left", fontsize=8.8, color=TEXT, fontweight="bold")
    trans = matplotlib.transforms.blended_transform_factory(fig.transFigure, ax.transData)
    for i, (metric, ml, rm, xl, rx) in enumerate(vals):
        yc = ymax - 0.5 - i
        bar(yc + 0.18, rm, BLUE)
        if rx is not None:
            bar(yc - 0.18, rx, ORANGE)
        else:
            ax.text(0.01, yc - 0.18, "no Match13 equivalent", va="center", ha="left", fontsize=8.2, color=MUTED, style="italic")
        fig.text(0.26 / W_IN, yc + 0.17, metric, transform=trans, fontsize=9.6, fontweight="bold", color=TEXT, va="center")
        fig.text(0.26 / W_IN, yc - 0.20, f"vs. {ml}" + (f"  ·  {xl}" if xl else ""), transform=trans,
                 fontsize=8.2, color=DIM, va="center")
    # reference
    yr = -0.62
    ax.plot([-(L_in - 0.26) / (W_IN - L_in - R_in), 1.0], [-0.12, -0.12], color=AXIS, lw=0.8, clip_on=False)
    bar(yr, ref, GRAY)
    fig.text(0.26 / W_IN, yr + 0.14, "Reference", transform=trans, fontsize=9.6, fontweight="bold", color=TEXT, va="center")
    fig.text(0.26 / W_IN, yr - 0.20, "scouted teleop fuel vs. teleop xP", transform=trans, fontsize=8.2, color=DIM, va="center")
    ax.set_yticks([])
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.25))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".") if v else "0"))
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)
    feed = rho("joined.cyc_per_min|feeding", "manta.feeding")[0]
    notes = ("1 = identical team ranking, 0 = unrelated. Manta bars use only the 702 team-matches both sources cover; "
             "Match13 = each team's xP after its last Curie qual. Match13 rates scoring, so it has nothing to compare "
             "defense, dead time or beaching against. *Sign flipped: more time moving should mean less dead time. "
             f"Teleop cycles vs. scouted feeding is {feed:+.2f}, as expected: a feeder never makes the trip home.")
    yy = (foot_in - 0.44) / H
    for ln in wrap_lines(notes, 120):
        fig.text(0.26 / W_IN, yy, ln, fontsize=8.6, color=DIM, va="top")
        yy -= 0.19 / H
    fig.savefig(f"{OUT}/cmp_summary.png", dpi=DPI)
    plt.close(fig)
    for v in vals:
        print(f"  {v[0]:22s} manta {v[2]:+.2f}  match13 {'' if v[4] is None else f'{v[4]:+.2f}'}")
    print(f"  reference {ref:+.2f}")


if "pairs" in sys.argv or len(sys.argv) == 1:
    pairs()
if "summary" in sys.argv or len(sys.argv) == 1:
    summary_grouped()
