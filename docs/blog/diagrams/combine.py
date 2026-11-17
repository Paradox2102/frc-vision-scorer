"""Stack blog images into combined figures (Chief Delphi upload limit). Run from docs/blog."""
import os
from PIL import Image

BG, GAP, W = (13, 18, 32), 28, 2000

def load(p):
    return Image.open(os.path.join('img', p)).convert('RGB')

def scale_w(im, w):
    return im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)

def row(ims, w=W, gap=GAP):
    """Side by side at a common height, filling width w."""
    ratios = [i.width / i.height for i in ims]
    h = round((w - gap * (len(ims) - 1)) / sum(ratios))
    out = Image.new('RGB', (w, h), BG); x = 0
    for i, r in zip(ims, ratios):
        iw = round(h * r); out.paste(i.resize((iw, h), Image.LANCZOS), (x, 0)); x += iw + gap
    return out

def stack(ims, w=W, gap=GAP):
    ims = [scale_w(i, w) for i in ims]
    out = Image.new('RGB', (w, sum(i.height for i in ims) + gap * (len(ims) - 1)), BG); y = 0
    for i in ims:
        out.paste(i, (0, y)); y += i.height + gap
    return out

def save(im, name):
    png = os.path.join('img', name + '.png')
    im.save(png, optimize=True)
    if os.path.getsize(png) > 3_500_000:          # Discourse-friendly size
        os.remove(png); jpg = os.path.join('img', name + '.jpg')
        im.save(jpg, quality=90, optimize=True); png = jpg
    print(f'{png}: {im.width}x{im.height}, {os.path.getsize(png)/1e6:.1f} MB')

save(stack([row([load('calib_main_2026cur.jpg'), load('calib_corner_2025hop.jpg')]), load('calib_events.png')]), 'calibration_combined')
save(stack([load('scouting_rankings_defense.png'), load('scouting_rankings_cycles.png')]), 'scouting_rankings_combined')
save(stack([load('scouting_heatmap_cur.png'), load('scouting_heatmap_defender.png')]), 'scouting_heatmaps_combined')
save(stack([load('cmp_auto.png'), load('cmp_cycles.png'), load('cmp_speed.png')], w=1380), 'cmp_auto_cycles_speed')
save(stack([load('cmp_defense.png'), load('cmp_bump_manta.png')], w=1380), 'cmp_defense_bump')
save(stack([load('gt_annotator.png'), load('diagram_testing.png')]), 'testing_combined')
