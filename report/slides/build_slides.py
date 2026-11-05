"""Build the 15-20 min team presentation from results.json and report/figures."""
import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / 'report' / 'figures'
OUT = Path(__file__).resolve().parent / 'kg_completion_presentation.pptx'
R = json.loads((ROOT / 'results.json').read_text(encoding='utf-8'))
MAIN = R['main_results']

NAVY = RGBColor(0x1F, 0x2A, 0x44)
TEAL = RGBColor(0x0F, 0xA3, 0xB1)
ORANGE = RGBColor(0xF1, 0x8F, 0x01)
RED = RGBColor(0xC0, 0x39, 0x2B)
INK = RGBColor(0x22, 0x26, 0x30)
MUTED = RGBColor(0x5B, 0x64, 0x72)
LIGHT = RGBColor(0xF4, 0xF6, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = 'Calibri'

SPK = {
    'cp': 'Charan Ponaganti',
    'mc': 'M Charan',
    'ld': 'Lalit Deshmane',
}
LABELS = {
    'distmult': 'DistMult', 'rgcn': 'R-GCN', 'compgcn_sub': 'CompGCN-sub',
    'compgcn_mult': 'CompGCN-mult', 'compgcn_corr': 'CompGCN-corr',
    'decoder_transe': 'TransE', 'decoder_complex': 'ComplEx',
}
ORDER = list(LABELS)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
W, H = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]
_n = [0]


def style(run, size, bold=False, color=INK, italic=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color


def textbox(slide, x, y, w, h, text, size=18, bold=False, color=INK,
            align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    lines = text if isinstance(text, list) else [text]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        style(p.add_run(), size, bold, color, italic)
        p.runs[0].text = line
    return tb


def bullets(slide, x, y, w, h, items, size=19, gap=8):
    """items: str or (str, level) or (bold_lead, rest)."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        level = 0
        lead, rest = None, it
        if isinstance(it, tuple):
            if isinstance(it[1], int):
                rest, level = it
            else:
                lead, rest = it
        indent = '      ' * level
        mark = '•  ' if level == 0 else '–  '
        r0 = p.add_run(); r0.text = indent + mark
        style(r0, size - 2 * level, color=TEAL if level == 0 else MUTED, bold=True)
        if lead:
            r1 = p.add_run(); r1.text = lead + ' '
            style(r1, size - 2 * level, bold=True, color=NAVY)
        r2 = p.add_run(); r2.text = rest
        style(r2, size - 2 * level, color=INK if level == 0 else MUTED)
    return tb


def rect(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line; s.line.width = Pt(1.25)
    s.shadow.inherit = False
    return s


def picture(slide, path, x, y, max_w, max_h):
    iw, ih = Image.open(path).size
    scale = min(max_w / iw, max_h / ih)
    w, h = int(iw * scale), int(ih * scale)
    return slide.shapes.add_picture(str(path), x + (max_w - w) // 2, y + (max_h - h) // 2, w, h)


def new_slide(title, speaker, notes, kicker=None):
    _n[0] += 1
    s = prs.slides.add_slide(BLANK)
    rect(s, 0, 0, W, Inches(1.05), NAVY)
    rect(s, 0, Inches(1.05), W, Inches(0.06), TEAL)
    textbox(s, Inches(0.55), Inches(0.12), Inches(10.5), Inches(0.85), title,
            size=30, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    if kicker:
        textbox(s, Inches(9.2), Inches(0.12), Inches(3.6), Inches(0.85), kicker,
                size=14, color=RGBColor(0xB8, 0xC4, 0xD6), align=PP_ALIGN.RIGHT,
                anchor=MSO_ANCHOR.MIDDLE)
    textbox(s, Inches(0.55), Inches(7.02), Inches(8), Inches(0.35),
            f'Speaker: {SPK[speaker]}', size=11, color=MUTED)
    textbox(s, Inches(11.8), Inches(7.02), Inches(1.0), Inches(0.35),
            str(_n[0]), size=11, color=MUTED, align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = f'[{SPK[speaker]}]\n{notes}'
    return s


def card(slide, x, y, w, h, head, body, color=TEAL, size=16):
    rect(slide, x, y, w, h, LIGHT)
    rect(slide, x, y, Inches(0.09), h, color)
    textbox(slide, x + Inches(0.25), y + Inches(0.12), w - Inches(0.35), Inches(0.5),
            head, size=19, bold=True, color=NAVY)
    bullets(slide, x + Inches(0.2), y + Inches(0.65), w - Inches(0.35), h - Inches(0.75),
            body, size=size, gap=5)


def table(slide, x, y, w, rows, col_w, size=15, highlight_row=None, row_h=0.42):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, x, y, w, Inches(row_h * nr)).table
    for j, cw in enumerate(col_w):
        gt.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        gt.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = gt.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.08)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb = NAVY
            elif i == highlight_row:
                c.fill.fore_color.rgb = RGBColor(0xD6, 0xF1, 0xF3)
            else:
                c.fill.fore_color.rgb = WHITE if i % 2 else LIGHT
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run(); r.text = str(val)
            style(r, size, bold=(i == 0 or i == highlight_row),
                  color=WHITE if i == 0 else INK)
    return gt


def f3(v):
    return f'{v:.3f}'


# 1 ─ Title ───────────────────────────────────────────────────────────────
_n[0] += 1
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, W, H, NAVY)
rect(s, 0, Inches(4.55), W, Inches(0.07), TEAL)
textbox(s, Inches(0.8), Inches(1.3), Inches(11.7), Inches(1.9),
        ['Knowledge Graph Completion', 'with R-GCN and CompGCN'],
        size=44, bold=True, color=WHITE)
textbox(s, Inches(0.8), Inches(3.35), Inches(11.7), Inches(0.9),
        'Link prediction on ogbl-biokg with a PySpark pipeline on a Kaggle T4',
        size=22, color=RGBColor(0xB8, 0xC4, 0xD6))
for i, (k, roll) in enumerate([('cp', '240150024'), ('mc', '240150018'), ('ld', '240150044')]):
    x = Inches(0.8 + i * 4.0)
    textbox(s, x, Inches(5.0), Inches(3.8), Inches(0.5), SPK[k], size=22, bold=True, color=WHITE)
    textbox(s, x, Inches(5.5), Inches(3.8), Inches(0.45), roll, size=16, color=TEAL)
textbox(s, Inches(0.8), Inches(6.6), Inches(11.7), Inches(0.45),
        'Big Data course project  ·  November 2026', size=14, color=RGBColor(0x8A, 0x97, 0xAD))
s.notes_slide.notes_text_frame.text = (
    f"[{SPK['cp']}] ~0:30\nIntroduce the team and the one-line goal: predict missing links in a "
    "large biomedical knowledge graph, using Spark for the data side and graph neural networks "
    "for the learning side, all within a free Kaggle T4 GPU.")

# 2 ─ Agenda ──────────────────────────────────────────────────────────────
s = new_slide('Roadmap', 'cp',
              "~0:45\nWalk through the three parts and who presents each. Total ~18 minutes plus Q&A.")
blocks = [
    ('cp', '1 · Problem & data', ['What KG completion is', 'Why it is a Big Data problem',
                                  'ogbl-biokg', 'PySpark ETL & scaling'], TEAL),
    ('mc', '2 · Models & training', ['DistMult, R-GCN, CompGCN', 'Fitting GNNs on a 16 GB T4',
                                     'Training recipe', 'Two evaluation protocols'], ORANGE),
    ('ld', '3 · Results & lessons', ['Main results (OGB + candidates)', 'Ablations & error analysis',
                                     'Engineering lessons', 'Conclusion & limitations'], NAVY),
]
for i, (k, head, items, col) in enumerate(blocks):
    x = Inches(0.55 + i * 4.15)
    rect(s, x, Inches(1.55), Inches(3.9), Inches(0.75), col)
    textbox(s, x + Inches(0.2), Inches(1.55), Inches(3.6), Inches(0.75), head,
            size=21, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    rect(s, x, Inches(2.3), Inches(3.9), Inches(3.75), LIGHT)
    bullets(s, x + Inches(0.2), Inches(2.5), Inches(3.55), Inches(3.0), items, size=18)
    textbox(s, x + Inches(0.2), Inches(5.45), Inches(3.6), Inches(0.5), SPK[k],
            size=15, bold=True, color=col, italic=True)
textbox(s, Inches(0.55), Inches(6.3), Inches(12.2), Inches(0.5),
        '≈ 6 minutes per speaker  ·  ~18 min total + Q&A', size=15, color=MUTED, align=PP_ALIGN.CENTER)

# 3 ─ Problem ─────────────────────────────────────────────────────────────
s = new_slide('The problem: completing a knowledge graph', 'cp',
              "~1:15\nA KG is a set of (head, relation, tail) facts. Real KGs are incomplete. Link "
              "prediction ranks every candidate entity for a query like (aspirin, treats, ?). "
              "We rank in both directions: missing tail and missing head. Metrics are MRR and Hits@k: "
              "how high the true answer lands in the ranked list.")
bullets(s, Inches(0.55), Inches(1.45), Inches(6.4), Inches(5.2), [
    ('Knowledge graph:', 'facts stored as triples (head, relation, tail).'),
    ('Problem:', 'real KGs are incomplete; many true links are missing.'),
    ('Task:', 'given (h, r, ?) or (?, r, t), rank all entities so the true one is near the top.'),
    ('Metrics:', 'MRR (mean reciprocal rank) and Hits@1/3/10.'),
    ('Our goal:', 'compare GNN encoders (R-GCN, CompGCN) against shallow embeddings on a real, large KG.'),
], size=19)
rect(s, Inches(7.3), Inches(1.5), Inches(5.5), Inches(4.9), LIGHT)
textbox(s, Inches(7.55), Inches(1.65), Inches(5.0), Inches(0.5), 'Example query (biomedical)',
        size=17, bold=True, color=NAVY)
nodes = [('Drug A', 7.7, 2.5, TEAL), ('Protein P', 10.4, 2.5, NAVY),
         ('Disease D', 7.7, 4.6, NAVY), ('?', 10.4, 4.6, ORANGE)]
for name, x, y, col in nodes:
    c = rect(s, Inches(x), Inches(y), Inches(2.1), Inches(0.85), col, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    tf = c.text_frame; tf.text = name
    style(tf.paragraphs[0].runs[0], 18, bold=True, color=WHITE)
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
for (x1, y1, x2, y2, lab, lx, ly) in [
        (9.8, 2.92, 10.4, 2.92, 'targets', 9.45, 2.05),
        (8.75, 3.35, 8.75, 4.6, 'treats', 8.85, 3.75),
        (11.45, 3.35, 11.45, 4.6, 'associated', 11.55, 3.75)]:
    ln = s.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    ln.line.color.rgb = MUTED; ln.line.width = Pt(2)
    textbox(s, Inches(lx), Inches(ly), Inches(1.4), Inches(0.4), lab, size=13, color=MUTED, italic=True)
textbox(s, Inches(7.55), Inches(5.65), Inches(5.0), Inches(0.6),
        'Query: (Protein P, associated, ?) → rank all 93k entities',
        size=15, color=INK, italic=True)

# 4 ─ Constraints ─────────────────────────────────────────────────────────
s = new_slide('Why this is a Big Data problem', 'cp',
              "~1:15\nTwo constraints shaped every decision. First, the course requires Spark for the "
              "heavy data work. Second, we only have a Kaggle T4: 16 GB VRAM, ~30 GB RAM, 12-hour "
              "sessions. Wikidata5M and wikikg2 have millions of entities; full-graph GNN training on them "
              "does not fit. ogbl-biokg is the largest OGB KG that does, with an official evaluator.")
rows = [['Dataset', 'Entities', 'Relations', 'Train triples', 'Fits T4 GNN?'],
        ['ogbl-biokg', '93,773', '51', '4.76M', 'Yes (sampled)'],
        ['ogbl-wikikg2', '2.50M', '535', '16.1M', 'No'],
        ['Wikidata5M', '4.59M', '822', '20.6M', 'No'],
        ['FB15k-237', '14,541', '237', '272k', 'Yes (too small)']]
table(s, Inches(0.55), Inches(1.5), Inches(7.6), rows, [1.9, 1.3, 1.25, 1.6, 1.55], size=15, highlight_row=1)
card(s, Inches(8.55), Inches(1.5), Inches(4.25), Inches(2.35), 'Constraint 1 · Spark', [
    'Cleaning, statistics, indexing, filter tables in PySpark',
    'Not a hidden pandas step'], color=TEAL)
card(s, Inches(8.55), Inches(4.05), Inches(4.25), Inches(2.6), 'Constraint 2 · Kaggle T4', [
    '16 GB VRAM · ~30 GB RAM', '12-hour session limit',
    'Full-batch GNN on millions of entities does not fit'], color=ORANGE)
textbox(s, Inches(0.55), Inches(4.0), Inches(7.6), Inches(1.2),
        'Decision: train on ogbl-biokg with neighbor-sampled mini-batches; keep Wikidata5M as a '
        'Spark-only subset option.', size=17, color=NAVY, bold=True)

# 5 ─ Dataset ─────────────────────────────────────────────────────────────
st = R['spark_stats']
s = new_slide('Dataset: ogbl-biokg', 'cp',
              "~1:00\nNumbers here all come from Spark aggregations. The degree distribution is very "
              "heavy-tailed: median 22 but a max of 62,733. That tail is why we cannot naively expand "
              "two-hop neighborhoods: one hub would blow up GPU memory. Every relation is many-to-many "
              "under the Bordes threshold, which matters later in error analysis.")
stats = [('93,773', 'entities'), ('51', 'relations'), (f"{st['n_triples']/1e6:.2f}M", 'train triples'),
         ('22', 'median degree'), ('62,733', 'max degree'), ('5', 'entity types')]
for i, (v, lab) in enumerate(stats):
    x = Inches(0.55 + (i % 2) * 2.55); y = Inches(1.5 + (i // 2) * 1.6)
    rect(s, x, y, Inches(2.4), Inches(1.4), LIGHT)
    textbox(s, x, y + Inches(0.12), Inches(2.4), Inches(0.7), v, size=28, bold=True, color=TEAL, align=PP_ALIGN.CENTER)
    textbox(s, x, y + Inches(0.8), Inches(2.4), Inches(0.45), lab, size=15, color=MUTED, align=PP_ALIGN.CENTER)
picture(s, FIG / 'fig_degree_loglog.png', Inches(5.8), Inches(1.4), Inches(7.0), Inches(4.7))
textbox(s, Inches(5.8), Inches(6.2), Inches(7.0), Inches(0.6),
        'Heavy-tailed degrees → hubs make full 2-hop expansion unsafe on a GPU',
        size=15, color=NAVY, italic=True, align=PP_ALIGN.CENTER)

# 6 ─ Spark pipeline ──────────────────────────────────────────────────────
s = new_slide('PySpark pipeline', 'cp',
              "~1:15\nSpark runs in local[*] mode with Arrow and Kryo, 12 GB driver. Stages: ingest with "
              "an explicit schema, clean duplicates and self-loops, profile degrees and relation types, "
              "build dense IDs and inverse edges for message passing, and build an int64 key for every "
              "true triple so filtered ranking can run on the GPU. Then spark.stop() frees RAM before "
              "training.")
stages = [('Ingest', 'explicit (h,r,t)\nschema'), ('Clean', 'nulls, duplicates,\nself-loops'),
          ('Profile', 'degrees, relation\ncategories'), ('Index', 'dense IDs +\ninverse edges'),
          ('Filter keys', 'int64 key per\ntrue triple'), ('Hand-off', 'tensors →\nspark.stop()')]
bw, gap = 1.85, 0.27
for i, (h, b) in enumerate(stages):
    x = Inches(0.55 + i * (bw + gap))
    col = TEAL if i < 5 else ORANGE
    rect(s, x, Inches(1.75), Inches(bw), Inches(0.7), col)
    textbox(s, x, Inches(1.75), Inches(bw), Inches(0.7), h, size=18, bold=True, color=WHITE,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    rect(s, x, Inches(2.45), Inches(bw), Inches(1.15), LIGHT)
    textbox(s, x, Inches(2.5), Inches(bw), Inches(1.05), b.split('\n'), size=14, color=INK,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    if i < len(stages) - 1:
        a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x + Inches(bw + 0.03), Inches(1.95),
                               Inches(gap - 0.06), Inches(0.3))
        a.fill.solid(); a.fill.fore_color.rgb = MUTED; a.line.fill.background()
bullets(s, Inches(0.55), Inches(4.0), Inches(12.2), Inches(2.8), [
    ('Config:', 'local[*], Arrow on, Kryo serializer, 12 GB driver, 8 shuffle partitions.'),
    ('Inverse edges:', '(t, r + |R|, h) → 102 message-passing relation types.'),
    ('Filtered ranking:', 'Spark builds one int64 key per true triple; the GPU does a sorted search.'),
    ('Memory safety:', 'Spark is stopped before any GPU training starts.'),
], size=18)

# 7 ─ Spark scaling ───────────────────────────────────────────────────────
s = new_slide('Spark scalability', 'cp',
              "~1:00\nWe timed count + degree + relation frequency on 10 to 100 percent of the data and "
              "on 2 to 16 partitions. Everything finishes in about two seconds. The 100 percent point "
              "is fastest only because that DataFrame was already cached; we report that honestly. "
              "Partition count is flat because one node saturates quickly. Hand over to M Charan.")
picture(s, FIG / 'fig_spark_scale.png', Inches(0.4), Inches(1.35), Inches(8.2), Inches(3.4))
sc = R['spark_scale']
rows = [['Setting', 'Seconds']] + [
    [f"{int(d['x']*100)}% data" if d['axis'] == 'fraction' else f"{int(d['x'])} partitions",
     f"{d['seconds']:.2f}"] for d in sc]
table(s, Inches(8.9), Inches(1.35), Inches(3.9), rows, [2.4, 1.5], size=13, row_h=0.36)
bullets(s, Inches(0.55), Inches(5.0), Inches(8.0), Inches(1.9), [
    'All of ETL profiling on 4.76M triples runs in ≈ 1–2 s on one node.',
    '100% is fastest because it is cached; not super-linear scaling.',
    'Partitions 2→16 are flat: a single node saturates quickly.',
], size=16, gap=5)

# 8 ─ Models ──────────────────────────────────────────────────────────────
s = new_slide('Models we compare', 'mc',
              "~1:30\nThe decoder is DistMult for every GNN, so differences come from the encoder. "
              "R-GCN uses relation-specific weights with basis decomposition so 102 relation types "
              "share 8 bases. CompGCN puts the relation inside the message using a composition: "
              "subtraction, multiplication, or circular correlation. As extra baselines we trained "
              "shallow TransE and ComplEx with the same budget.")
card(s, Inches(0.55), Inches(1.45), Inches(3.95), Inches(4.0), 'DistMult (decoder)', [
    'score(h, r, t) = Σ hₖ · rₖ · tₖ', 'Shared decoder for all GNNs',
    'Also a shallow baseline'], color=MUTED)
card(s, Inches(4.7), Inches(1.45), Inches(3.95), Inches(4.0), 'R-GCN encoder', [
    'Per-relation weight Wᵣ + self-loop', 'Basis decomposition: Wᵣ = Σ aᵣᵦ Vᵦ',
    '8 bases shared by 102 relations', '2 layers, dim 128'], color=TEAL)
card(s, Inches(8.85), Inches(1.45), Inches(3.95), Inches(4.0), 'CompGCN encoder', [
    'Message = W · φ(neighbor, relation)', 'sub: h − r', 'mult: h ⊙ r',
    'corr: h ⋆ r (FFT)', 'Relations updated each layer'], color=ORANGE)
rect(s, Inches(0.55), Inches(5.7), Inches(12.25), Inches(0.95), LIGHT)
textbox(s, Inches(0.8), Inches(5.75), Inches(11.8), Inches(0.85),
        'Extra shallow baselines (same budget): TransE  (−‖h + r − t‖₁)   and   ComplEx  (complex bilinear)',
        size=17, color=NAVY, bold=True, anchor=MSO_ANCHOR.MIDDLE)

# 9 ─ Fitting on T4 ───────────────────────────────────────────────────────
s = new_slide('Making GNNs fit on a 16 GB T4', 'mc',
              "~1:30\nEncoding all 94k entities with message passing every step does not fit. So we "
              "encode only the query entity: sample up to 10 neighbors per hop for two hops, run the "
              "GNN on that small subgraph, and add the base entity vector as a residual. Candidates are "
              "scored from the global embedding table. Mixed precision for speed; CompGCN composition "
              "is forced to float32 because half-precision FFT broke correlation. OOM halves batch and fanout.")
flow = [('Query entity', TEAL), ('Sample 2-hop\nfanout [10, 10]', NAVY), ('GNN on\nsubgraph', NAVY),
        ('+ residual\nbase embedding', NAVY), ('DistMult score vs\ncandidate table', ORANGE)]
bw, gap = 2.15, 0.35
for i, (t, col) in enumerate(flow):
    x = Inches(0.55 + i * (bw + gap))
    b = rect(s, x, Inches(1.6), Inches(bw), Inches(1.15), col, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    textbox(s, x, Inches(1.6), Inches(bw), Inches(1.15), t.split('\n'), size=16, bold=True,
            color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    if i < len(flow) - 1:
        a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x + Inches(bw + 0.05), Inches(2.0),
                               Inches(gap - 0.1), Inches(0.35))
        a.fill.solid(); a.fill.fore_color.rgb = MUTED; a.line.fill.background()
bullets(s, Inches(0.55), Inches(3.2), Inches(6.1), Inches(3.6), [
    ('Query-side encoding:', 'only the query gets message passing; candidates come from model.ent.'),
    ('Residual:', 'GNN output + base embedding keeps queries and candidates in one space.'),
    ('Neighbor sampling:', 'bounded cost even around 62k-degree hubs.'),
], size=17)
bullets(s, Inches(6.9), Inches(3.2), Inches(5.9), Inches(3.6), [
    ('Mixed precision', 'for speed; CompGCN composition kept in float32.'),
    ('OOM fallback:', 'halve batch and fanout, retry.'),
    ('Fail-soft:', 'each model saved separately; resume after Kaggle restarts.'),
], size=17)

# 10 ─ Training recipe ────────────────────────────────────────────────────
b = R['budget']
s = new_slide('Training recipe', 'mc',
              "~1:15\nEvery step trains both directions: predict the tail from the head and the head "
              "from the tail. All seven models get the same budget, 8 epochs of 400 steps with 16 "
              "negatives, so the comparison is fair. Negatives are type-aware: a drug is replaced by "
              "another drug. Self-adversarial weighting from RotatE focuses the loss on hard negatives.")
bullets(s, Inches(0.55), Inches(1.45), Inches(6.6), Inches(5.3), [
    ('Bidirectional:', 'each step trains (h, r, ?) and (?, r, t).'),
    ('Matched budget:', 'all 7 models get identical epochs, steps, and negatives.'),
    ('Type-aware negatives:', 'corrupt a drug with another drug, not a random protein.'),
    ('Self-adversarial loss:', 'harder negatives get more weight (RotatE-style).'),
    ('Early stopping:', 'on validation MRR, patience 5; keep the best checkpoint.'),
], size=18)
rows = [['Setting', 'Value'],
        ['Epochs × steps', f"{b['gnn']['epochs']} × {b['gnn']['steps']}"],
        ['Negatives / batch', f"{b['gnn']['neg']} / {b['batch']}"],
        ['Embedding dim', str(b['dim'])],
        ['Layers / R-GCN bases', f"{b['layers']} / {b['bases']}"],
        ['Fanout', '[10, 10]'],
        ['Optimizer', 'Adam 1e-3, clip 1.0'],
        ['Wall clock (all)', f"{R['wall_clock_h']:.2f} h on T4"]]
table(s, Inches(7.5), Inches(1.5), Inches(5.3), rows, [2.8, 2.5], size=15)

# 11 ─ Protocols ──────────────────────────────────────────────────────────
s = new_slide('Two evaluation protocols', 'mc',
              "~1:15\nWe report two views. The candidate protocol ranks the answer among 500 random "
              "entities, filtering other true answers using the Spark keys. The official OGB evaluator "
              "uses OGB's own type-constrained negatives over 162,870 test triples; that is the number "
              "to compare with the literature. One lesson: OGB negatives are row-aligned with OGB's test "
              "order, and Spark reorders rows, so we evaluate OGB on the original order. Hand over to Lalit.")
card(s, Inches(0.55), Inches(1.5), Inches(5.95), Inches(3.6), 'Filtered candidates (ours)', [
    'True entity vs 500 random entities', 'Other true answers masked via Spark keys',
    'Head + tail, 8,000 test triples', 'Fast; used for validation each epoch'], color=TEAL, size=17)
card(s, Inches(6.85), Inches(1.5), Inches(5.95), Inches(3.6), 'Official OGB evaluator', [
    "OGB's type-constrained negatives", f"All {R['eval']['ogb_test_triples']:,} test triples, head + tail",
    'Comparable to published numbers', 'Main metric in the report'], color=ORANGE, size=17)
rect(s, Inches(0.55), Inches(5.4), Inches(12.25), Inches(1.3), RGBColor(0xFD, 0xF0, 0xE0))
textbox(s, Inches(0.8), Inches(5.45), Inches(11.8), Inches(1.2), [
    'Lesson learned: OGB negatives are aligned to OGB\'s test order. Spark reorders rows,',
    'so an early version paired the wrong negatives. We now evaluate OGB on the original order.'],
    size=16, color=INK, anchor=MSO_ANCHOR.MIDDLE)

# 12 ─ OGB results ────────────────────────────────────────────────────────
s = new_slide('Main result: official OGB evaluator', 'ld',
              "~1:30\nThis is the headline table. CompGCN with multiplication is best on every metric: "
              "MRR 0.513, Hits@10 0.736. CompGCN-sub is close behind, then R-GCN at 0.401. Shallow "
              "TransE and ComplEx sit around 0.26 and DistMult is lowest at 0.149. CompGCN-corr failed "
              "to learn. Message passing clearly helps under type-constrained negatives.",
              kicker='162,870 test triples')
rows = [['Model', 'MRR', 'H@1', 'H@3', 'H@10']]
for k in ORDER:
    o = MAIN[k]['ogb']
    rows.append([LABELS[k], f3(o['mrr_list']), f3(o['hits@1_list']), f3(o['hits@3_list']), f3(o['hits@10_list'])])
table(s, Inches(0.55), Inches(1.45), Inches(6.2), rows, [2.2, 1.0, 1.0, 1.0, 1.0], size=15,
      highlight_row=1 + ORDER.index('compgcn_mult'), row_h=0.5)
cd = CategoryChartData()
cd.categories = [LABELS[k] for k in ORDER]
cd.add_series('MRR', [round(MAIN[k]['ogb']['mrr_list'], 3) for k in ORDER])
cd.add_series('Hits@10', [round(MAIN[k]['ogb']['hits@10_list'], 3) for k in ORDER])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(7.0), Inches(1.35),
                        Inches(5.9), Inches(4.4), cd)
ch = gf.chart
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False
ch.legend.font.size = Pt(13)
ch.value_axis.maximum_scale = 0.8; ch.value_axis.has_major_gridlines = True
ch.value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xDD, 0xE1, 0xE6)
ch.value_axis.format.line.fill.background()
ch.value_axis.tick_labels.font.size = Pt(12)
ch.category_axis.tick_labels.font.size = Pt(11)
for ser, col in zip(ch.series, [TEAL, ORANGE]):
    ser.format.fill.solid(); ser.format.fill.fore_color.rgb = col
rect(s, Inches(7.0), Inches(5.9), Inches(5.8), Inches(0.85), LIGHT)
textbox(s, Inches(7.2), Inches(5.9), Inches(5.5), Inches(0.85),
        'CompGCN-mult > CompGCN-sub > R-GCN > TransE/ComplEx > DistMult',
        size=15, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)

# 13 ─ Candidate results ──────────────────────────────────────────────────
s = new_slide('Second protocol agrees', 'ld',
              "~1:00\nUnder the 500-candidate protocol the ranking is the same: CompGCN-mult on top with "
              "MRR 0.526 and Hits@10 0.80. Absolute numbers are higher because random candidates are "
              "easier than OGB's type-constrained ones. In an earlier run with an unequal budget DistMult "
              "looked best here; with a matched budget that disappears.",
              kicker='8,000 test triples · 500 candidates')
rows = [['Model', 'MRR', 'H@10', 'MR']]
for k in ORDER:
    m = MAIN[k]
    rows.append([LABELS[k], f3(m['MRR']), f3(m['Hits@10']), f"{m['MR']:.1f}"])
table(s, Inches(0.55), Inches(1.45), Inches(5.4), rows, [2.2, 1.05, 1.05, 1.1], size=15,
      highlight_row=1 + ORDER.index('compgcn_mult'), row_h=0.5)
picture(s, FIG / 'fig_main_bars.png', Inches(6.2), Inches(1.35), Inches(6.7), Inches(4.3))
bullets(s, Inches(6.3), Inches(5.75), Inches(6.5), Inches(1.1), [
    'Same ranking as OGB; numbers higher because random negatives are easier.',
    'Matched budget removed the earlier "DistMult wins" artifact.',
], size=15, gap=4)

# 14 ─ Curves ─────────────────────────────────────────────────────────────
s = new_slide('Training dynamics', 'ld',
              "~1:00\nLeft: loss. CompGCN-mult reaches the lowest loss, then CompGCN-sub and R-GCN. "
              "TransE plateaus highest. Right: validation MRR on 500 random candidates, which is why "
              "every healthy model sits near 0.8. CompGCN-corr starts near 0.2, collapses to almost "
              "zero, and early stopping ends it at epoch 6. All other models run the full 8 epochs.")
picture(s, FIG / 'fig_train_curves.png', Inches(0.55), Inches(1.35), Inches(12.2), Inches(4.4))
bullets(s, Inches(0.55), Inches(5.85), Inches(12.2), Inches(1.0), [
    ('Lowest loss:', 'CompGCN-mult, then CompGCN-sub and R-GCN; TransE plateaus highest.'),
    ('CompGCN-corr collapses', 'after epoch 1 even in float32; early-stopped at epoch 6 (all others run 8).'),
], size=16, gap=4)

# 15 ─ Ablations ──────────────────────────────────────────────────────────
ab = R['ablation']
s = new_slide('R-GCN ablations', 'ld',
              "~1:00\nHalf-length runs of R-GCN: 4 epochs of 200 steps. Changes are small. Four bases "
              "instead of eight is slightly best, one layer and eight negatives are about equal to the "
              "base, and halving the dimension to 64 hurts most. So capacity matters more than depth "
              "at this budget.", kicker='4 epochs × 200 steps')
names = {'base': 'Base (dim 128, 2 layers, 8 bases, 16 neg)', 'layers1': '1 layer',
         'neg8': '8 negatives', 'bases4': '4 bases', 'dim64': 'dim 64'}
rows = [['Setting', 'MRR', 'H@1', 'H@10', 'MR']]
for a in ab:
    rows.append([names.get(a['tag'], a['tag']), f3(a['MRR']), f3(a['Hits@1']), f3(a['Hits@10']), f"{a['MR']:.1f}"])
best = 1 + max(range(len(ab)), key=lambda i: ab[i]['MRR'])
table(s, Inches(0.55), Inches(1.5), Inches(8.0), rows, [4.0, 1.0, 1.0, 1.0, 1.0], size=15,
      highlight_row=best, row_h=0.55)
card(s, Inches(8.9), Inches(1.5), Inches(3.9), Inches(3.3), 'Takeaways', [
    'All within ≈ 0.03 MRR', '4 bases slightly best', 'dim 64 hurts most',
    'Depth 1 ≈ depth 2 here'], color=TEAL, size=16)

# 16 ─ Error analysis ─────────────────────────────────────────────────────
ea = R['error_analysis']['mrr_by_degree']
s = new_slide('Error analysis: rare entities are hard', 'ld',
              "~1:15\nAll relations are many-to-many, so a relation-category breakdown has only one bar. "
              "Degree is the informative slice: for CompGCN-mult, entities in the lowest degree quartile "
              "get MRR 0.18, while hubs get 0.55. Sparse entities have little neighborhood for the GNN "
              "to use. The embedding projection on the right shows loose clusters by entity type.")
picture(s, FIG / 'fig_mrr_by_degree.png', Inches(0.4), Inches(1.35), Inches(6.4), Inches(4.0))
picture(s, FIG / 'fig_umap_entities.png', Inches(7.0), Inches(1.35), Inches(5.8), Inches(4.0))
vals = list(ea.values())
bullets(s, Inches(0.55), Inches(5.5), Inches(12.2), Inches(1.4), [
    ('By degree (CompGCN-mult):', f'MRR {vals[0]:.2f} for the rarest quartile vs {vals[-1]:.2f} for hubs.'),
    ('By relation type:', 'all 51 relations are N–N, so that breakdown is uninformative.'),
], size=16, gap=4)

# 17 ─ Lessons ────────────────────────────────────────────────────────────
s = new_slide('Engineering lessons', 'ld',
              "~1:15\nFour real bugs we hit and fixed. Each is a general lesson for running graph learning "
              "on shared notebooks: check alignment of external negatives, watch numerical precision, "
              "avoid per-row Python loops in samplers, and never compile CUDA extensions on Kaggle.")
lessons = [
    ('OGB misalignment', 'Spark reordered test rows; OGB negatives are row-aligned. Fixed by '
                         'evaluating on the original order.', RED),
    ('Half-precision FFT', 'CompGCN-corr broke under AMP. Composition now runs in float32.', ORANGE),
    ('Kernel deaths (RAM)', 'Per-row negative sampling made millions of tiny tensors. Vectorized by type.', TEAL),
    ('CUDA 13 installs', 'Compiling torch-scatter killed the kernel. Binary wheels only, else skip.', NAVY),
]
for i, (h, t, col) in enumerate(lessons):
    x = Inches(0.55 + (i % 2) * 6.2); y = Inches(1.5 + (i // 2) * 2.65)
    rect(s, x, y, Inches(5.95), Inches(2.4), LIGHT)
    rect(s, x, y, Inches(0.1), Inches(2.4), col)
    textbox(s, x + Inches(0.3), y + Inches(0.15), Inches(5.5), Inches(0.5), h, size=20, bold=True, color=NAVY)
    textbox(s, x + Inches(0.3), y + Inches(0.75), Inches(5.5), Inches(1.5), t, size=16, color=INK)

# 18 ─ Conclusion ─────────────────────────────────────────────────────────
s = new_slide('Conclusion, limitations, future work', 'ld',
              "~1:15\nSummarize: Spark handles the data, sampled GNNs handle learning on a T4, and "
              "CompGCN-mult wins on both protocols under a fair budget. Be upfront about limits: our "
              "numbers are below long-schedule leaderboard results, candidates are not GNN-encoded, and "
              "Spark ran on one node. Then invite questions.")
card(s, Inches(0.55), Inches(1.45), Inches(4.0), Inches(5.3), 'Conclusions', [
    'Spark ETL on 4.76M triples in seconds',
    'Sampled GNNs fit a 16 GB T4',
    'CompGCN-mult best: OGB MRR 0.513',
    'GNNs ≫ shallow baselines at equal budget'], color=TEAL, size=16)
card(s, Inches(4.7), Inches(1.45), Inches(4.0), Inches(5.3), 'Limitations', [
    'Fast budget; below published SOTA',
    'Candidates not GNN-encoded',
    'CompGCN-corr unstable',
    'Spark on a single node'], color=ORANGE, size=16)
card(s, Inches(8.85), Inches(1.45), Inches(3.95), Inches(5.3), 'Future work', [
    'Encode both query and candidate',
    'ConvE decoder on the same encoder',
    'Longer schedule / stabilize corr',
    'Multi-node Spark; Wikidata5M training'], color=NAVY, size=16)

# 19 ─ Q&A ───────────────────────────────────────────────────────────────
_n[0] += 1
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, W, H, NAVY)
rect(s, 0, Inches(4.2), W, Inches(0.07), TEAL)
textbox(s, Inches(0.8), Inches(2.0), Inches(11.7), Inches(1.4), 'Thank you · Questions?',
        size=48, bold=True, color=WHITE)
textbox(s, Inches(0.8), Inches(4.6), Inches(11.7), Inches(0.6),
        'Code, notebook, results and report:', size=18, color=RGBColor(0xB8, 0xC4, 0xD6))
textbox(s, Inches(0.8), Inches(5.15), Inches(11.7), Inches(0.6),
        'github.com/Charanponaganti/Knowledge-Graph-Completion-with-R-GCN-CompGCN',
        size=18, bold=True, color=TEAL)
textbox(s, Inches(0.8), Inches(6.3), Inches(11.7), Inches(0.5),
        'Charan Ponaganti  ·  M Charan  ·  Lalit Deshmane', size=16, color=RGBColor(0x8A, 0x97, 0xAD))
s.notes_slide.notes_text_frame.text = (
    "[All] Q&A. Likely questions: why not Wikidata5M (memory), why DistMult decoder (assignment "
    "+ isolates encoder), why corr fails (numerical/optimization; reported honestly), why two protocols "
    "(OGB is comparable; candidates are fast for validation), what Spark adds (relational ETL + filter keys).")

prs.save(OUT)
print('saved', OUT, 'slides:', len(prs.slides))
