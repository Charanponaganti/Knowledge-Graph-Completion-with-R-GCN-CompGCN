"""Build the ACL report from reviewed prose and the repository's saved metrics.

Run with python-docx installed. The user's ACL template is an input, not modified.
PDF export is done separately through Microsoft Word on Windows.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import json
import re
import zipfile
from docx import Document
from docx.shared import Inches, Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'report'
DATA = json.loads((ROOT / 'results.json').read_text())
NAMES = {'distmult': 'DistMult', 'rgcn': 'R-GCN', 'compgcn_sub': 'CompGCN-sub',
         'compgcn_mult': 'CompGCN-mult', 'compgcn_corr': 'CompGCN-corr',
         'decoder_transe': 'TransE', 'decoder_complex': 'ComplEx'}

def degree_label(interval):
    lower, upper = (float(x.strip()) for x in interval.strip('()[]').split(','))
    return f'1 to {upper:,.0f}' if lower < 1 else f'Above {lower:,.0f} to {upper:,.0f}'

def tables():
    m = DATA['main_results']
    budget = DATA['budget']
    evaluation = DATA['eval']
    n_eval = evaluation['max_eval_triples']
    updates = budget['gnn']['epochs'] * budget['gnn']['steps']
    abl_updates = DATA['ablation'][0]['epochs'] * DATA['ablation'][0]['steps']
    result = {}
    result['dataset'] = ('Table 1: Prepared BioKG training graph. Counts are from the saved Spark statistics.',
        ['Statistic', 'Value'], [
        ['Entities', '93,773'], ['Original relations', '51'], ['Training triples', '4,762,676'],
        ['Rows removed in cleaning', '2'], ['Message relation identifiers', '102'],
        ['Minimum / median degree', '1 / 22'], ['Mean degree', '101.58'],
        ['99th percentile degree', '1,279.28'], ['Maximum degree', '62,733'],
        ['Many-to-many relations', '51']])
    result['config'] = ('Table 2: Main-run settings from the saved configuration. Epochs are fixed sampled intervals.',
        ['Setting', 'Value'], [
        ['Embedding coordinates', '128'], ['GNN layers / R-GCN bases', '2 / 8'],
        ['Batch size', '512'], ['Fanout per hop', '10, 10'],
        ['Maximum epochs × steps', f"{budget['gnn']['epochs']} × {budget['gnn']['steps']}"], ['Negatives per direction', '16'],
        ['Learning rate', '0.001'], ['Weight decay', '0.00001'],
        ['Dropout / gradient clip', '0.2 / 1.0'], ['Local test triples', f'{n_eval:,}'],
        ['Local sampled negatives', '500'], ['OGB test triples', '162,870']])
    result['local'] = (f'Table 3: Local filtered sampled-candidate results on {n_eval:,} triples, evaluated in both directions.',
        ['Model', 'MRR', 'H@1', 'H@3', 'H@10'],
        [[NAMES[k]] + [f'{v[x]:.4f}' for x in ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']] for k,v in m.items()])
    result['ogb'] = ('Table 4: OGB evaluator results on all 162,870 original test triples, with head and tail queries.',
        ['Model', 'MRR', 'H@1', 'H@3', 'H@10'],
        [[NAMES[k]] + [f'{v["ogb"][x]:.4f}' for x in ['mrr_list', 'hits@1_list', 'hits@3_list', 'hits@10_list']] for k,v in m.items()])
    result['direction'] = ('Table 5: Directional MRR in the local test. Gap is tail MRR minus head MRR.',
        ['Model', 'Tail', 'Head', 'Gap'],
        [[NAMES[k], f'{v["MRR_tail"]:.4f}', f'{v["MRR_head"]:.4f}', f'{v["MRR_tail"]-v["MRR_head"]:.4f}'] for k,v in m.items()])
    tags = {'base':'Short-budget base', 'layers1':'One layer', 'neg8':'Eight negatives', 'bases4':'Four bases', 'dim64':'Dimension 64'}
    result['ablation'] = (f'Table 6: R-GCN ablations with {abl_updates:,} configured updates each. The main schedule allows {updates:,} updates.',
        ['Setting', 'MRR', 'H@1', 'H@10'],
        [[tags[v['tag']]] + [f'{v[x]:.4f}' for x in ['MRR','Hits@1','Hits@10']] for v in DATA['ablation']])
    result['degree'] = ('Table 7: Saved reciprocal-mean-rank diagnostic by query-head degree. These values are not standard MRR.',
        ['Head occurrence degree', 'Diagnostic'],
        [[degree_label(interval), f'{v:.4f}'] for interval,v in DATA['error_analysis']['mrr_by_degree'].items()])
    result['spark'] = ('Table 8: Single-run local Spark timings. Fraction and partition sweeps time different workloads.',
        ['Setting', 'Approx. rows', 'Seconds'],
        [[f'{int(v["x"]*100)}% fraction' if v['axis']=='fraction' else f'{int(v["x"])} partitions',f'{v["approx_triples"]:,}',f'{v["seconds"]:.2f}'] for v in DATA['spark_scale']])
    return result

def num_off(p):
    n=OxmlElement('w:numPr'); i=OxmlElement('w:numId'); i.set(qn('w:val'),'0'); n.append(i)
    p._p.get_or_add_pPr().append(n)

def plain(p, text):
    # Preserve paragraph/frame properties while replacing template prose.
    for child in list(p._p):
        if child.tag != qn('w:pPr'): p._p.remove(child)
    p.add_run(text)

def texescape(s):
    replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}',
                  '×':r'$\times$','Σ':r'$\Sigma$','⊙':r'$\odot$','−':'-','^':r'\textasciicircum{}'}
    return ''.join(replacements.get(c,c) for c in s)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--template',type=Path,default=Path(r'C:\Users\Charan\Downloads\acl2023.docx')); args=ap.parse_args()
    template=args.template
    doc=Document(template)
    title=deepcopy(doc.tables[0]._tbl)
    sect=deepcopy(doc.sections[0]._sectPr)
    # Keep the supplied two-column page system, title frame and page footer.
    # Review line numbers are omitted for this named course report.
    for e in sect.findall(qn('w:lnNumType')): sect.remove(e)
    body=doc._element.body
    for e in list(body): body.remove(e)
    body.append(title); body.append(sect)
    title_table=doc.tables[0]
    for row in title_table.rows:
        for cell in row.cells:
            for p in cell.paragraphs: plain(p,'')
    p=title_table.cell(0,0).paragraphs[0]
    plain(p,'Knowledge Graph Completion\nwith R-GCN and CompGCN')
    # The template's used title paragraph has direct formatting, whereas its
    # unused ACL Title style adds a visible box. Reuse the visible pattern.
    p.style='ACL Address'
    for r in p.runs: r.font.name='Times New Roman'; r.font.size=Pt(15); r.bold=True
    p=title_table.cell(2,0).paragraphs[0]
    plain(p,'Charan Ponaganti (240150024)\nM Charan (240150018)    Lalit Deshmane (240150044)')
    p.style='ACL Address'
    for r in p.runs: r.font.size=Pt(12); r.bold=True
    # Source styles are retained, including ACL 11pt body and 10pt abstract.
    blocks=(REPORT/'report_content.md').read_text(encoding='utf-8').strip().split('\n\n')
    tmap=tables(); reference_mode=False; abstract=False
    tex=[r'\documentclass[11pt]{article}',r'\usepackage[preprint]{acl}',r'\usepackage{times,latexsym}',
         r'\usepackage[T1]{fontenc}',r'\usepackage[utf8]{inputenc}',r'\usepackage{microtype,graphicx,booktabs,amsmath,url}',
         r'\graphicspath{{../figures/}}',r'\title{Knowledge Graph Completion with R-GCN and CompGCN}',
         r'\author{Charan Ponaganti \\ 240150024 \And M Charan \\ 240150018 \And Lalit Deshmane \\ 240150044}',
         r'\begin{document}',r'\maketitle']
    for block in blocks:
        if block.startswith('# '): continue
        if block.startswith('## '):
            text=block[3:]
            abstract=text=='Abstract'; reference_mode=text=='References'
            style='ACL Abstract Heading' if abstract else 'ACL References Header' if reference_mode else 'ACL Section'
            p=doc.add_paragraph(text,style); num_off(p); p.paragraph_format.keep_with_next=True
            if not abstract: p.alignment=0
            p.paragraph_format.space_before=Pt(10 if not abstract else 0)
            if abstract: tex.append(r'\begin{abstract}')
            else: tex.append(r'\section*{'+texescape(text)+'}')
        elif block.startswith('### '):
            text=block[4:]; p=doc.add_paragraph(text,'ACL Subsection'); num_off(p); p.paragraph_format.keep_with_next=True
            p.alignment=0
            tex.append(r'\subsection*{'+texescape(text)+'}')
        elif block.startswith('[[TABLE:'):
            name=block[8:-2]; caption, headers, rows=tmap[name]
            t=doc.add_table(rows=1,cols=len(headers)); t.autofit=False
            width=3.03
            widths=[1.08]+[(width-1.08)/(len(headers)-1)]*(len(headers)-1) if len(headers)>=4 else ([1.82,1.21] if len(headers)==2 else [1.1,1.12,.81])
            for c,w in zip(t.columns,widths): c.width=Inches(w)
            for j,h in enumerate(headers): t.rows[0].cells[j].text=h
            for row in rows:
                cells=t.add_row().cells
                for j,value in enumerate(row): cells[j].text=value
            for ri,row in enumerate(t.rows):
                trpr=row._tr.get_or_add_trPr(); ns=OxmlElement('w:cantSplit'); trpr.append(ns)
                if ri==0: trpr.append(OxmlElement('w:tblHeader'))
                for j,cell in enumerate(row.cells):
                    cell.width=Inches(widths[j])
                    for p in cell.paragraphs:
                        p.style='ACL Text'; p.paragraph_format.first_line_indent=Pt(0)
                        p.paragraph_format.space_before=Pt(2); p.paragraph_format.space_after=Pt(2)
                        p.paragraph_format.keep_with_next=True
                        p.alignment=0 if j==0 else 2
                        for r in p.runs: r.font.size=Pt(9); r.bold=(ri==0)
                    pr=cell._tc.get_or_add_tcPr(); borders=OxmlElement('w:tcBorders')
                    if ri in (0,len(t.rows)-1):
                        for edge in (['top','bottom'] if ri==0 else ['bottom']):
                            e=OxmlElement('w:'+edge); e.set(qn('w:val'),'single'); e.set(qn('w:sz'),'4'); borders.append(e)
                    pr.append(borders)
            p=doc.add_paragraph(caption,'ACL Text'); p.paragraph_format.keep_with_next=False
            p.paragraph_format.keep_together=True
            p.paragraph_format.space_before=Pt(4)
            p.alignment=0
            for r in p.runs: r.font.size=Pt(10)
            p.paragraph_format.space_after=Pt(7)
            tex.extend([r'\begin{table}[t]\centering\small',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{l'+'r'*(len(headers)-1)+'}',r'\toprule',
                        ' & '.join(map(texescape,headers))+r' \\ \midrule'])
            tex.extend(' & '.join(map(texescape,row))+r' \\' for row in rows)
            tex.extend([r'\bottomrule\end{tabular}',r'\caption{'+texescape(caption.split(': ',1)[1])+r'}\end{table}'])
        elif block.startswith('[[FIG:'):
            name,caption=block[6:-2].split('|',1)
            p=doc.add_paragraph(); p.paragraph_format.keep_with_next=True
            p.add_run().add_picture(str(REPORT/'figures'/name),width=Inches(3.03))
            p=doc.add_paragraph(caption,'ACL Text'); p.paragraph_format.space_after=Pt(7)
            p.paragraph_format.keep_together=True
            p.paragraph_format.space_before=Pt(4)
            p.alignment=0
            for r in p.runs: r.font.size=Pt(10)
            tex.extend([r'\begin{figure}[t]\centering',r'\includegraphics[width=\columnwidth]{'+name+'}',r'\caption{'+texescape(caption.split(': ',1)[1])+r'}\end{figure}'])
        else:
            text=' '.join(block.splitlines())
            p=doc.add_paragraph(text,'ACL Abstract Text' if abstract else 'ACL References Text' if reference_mode else 'ACL Text First Line')
            p.paragraph_format.widow_control=True
            # Avoid Word expanding long URL segments beyond a column.
            if reference_mode: p.paragraph_format.keep_together=True
            tex.append(texescape(text)+'\n')
            if abstract: tex.append(r'\end{abstract}'); abstract=False
    tex.append(r'\end{document}')
    doc.core_properties.title='Knowledge Graph Completion with R-GCN and CompGCN'
    doc.core_properties.author='Charan Ponaganti; M Charan; Lalit Deshmane'
    doc.core_properties.subject='BioKG link prediction project report'
    out=REPORT/'Knowledge_Graph_Completion_Report.docx'
    doc.save(out)
    # Preserve every untouched package part byte-for-byte from the reference.
    # Only body, relationships, core metadata and types are authored.
    editable={'word/document.xml','word/_rels/document.xml.rels','docProps/core.xml','[Content_Types].xml'}
    with zipfile.ZipFile(template) as z: source={i.filename:z.read(i.filename) for i in z.infolist()}
    with zipfile.ZipFile(out) as z: final={i.filename:z.read(i.filename) for i in z.infolist()}
    for name,content in source.items():
        if name not in editable: final[name]=content
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for name,content in final.items(): z.writestr(name,content)
    (REPORT/'latex'/'main.tex').write_text('\n'.join(tex),encoding='utf-8')
    print('DOCX:',out)
    print('Narrative words:',len((REPORT/'report_content.md').read_text(encoding='utf-8').split()))
    print('Template SHA256:',hashlib.sha256(template.read_bytes()).hexdigest())

if __name__=='__main__': main()
