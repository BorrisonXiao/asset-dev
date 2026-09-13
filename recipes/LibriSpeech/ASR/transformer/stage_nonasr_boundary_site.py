#!/usr/bin/env python3
"""Stage the new analysis sub-tab in a copy of the existing Pages site.

This command does not commit or push. Validate its output before copying the
explicit changed files to the publication repository.
"""
import argparse
import json
from pathlib import Path
import re
import shutil

VERSION = "20260913-nonasr-boundaries"
SLUG = "nonasr-task-boundaries"
ROOT = Path(__file__).resolve().parents[4]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, default=ROOT / 'artifacts/segmenter/nonasr_boundary_audit_2026-09-12/html/nonasr-task-boundaries-standalone.html')
    parser.add_argument('--training-report', type=Path, default=ROOT / 'artifacts/segmenter/report.html')
    args = parser.parse_args()
    if args.site.resolve() == args.output.resolve():
        raise ValueError('Stage into a separate directory before publication')
    shutil.copytree(args.site, args.output, dirs_exist_ok=True, ignore=shutil.ignore_patterns('.git'))
    research = args.output / 'research'
    shutil.copyfile(args.report, research / args.report.name)
    asset_dir = args.report.parent / f'{SLUG}-assets'
    shutil.copytree(asset_dir, research / asset_dir.name, dirs_exist_ok=True)
    wrapper = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Single-task results and shared-audio boundary analysis on CREMA-D, Expresso and LibriSpeech.">
  <title>Non-ASR boundaries · Analyses · JSALT 2026</title>
  <link rel="stylesheet" href="../assets/site.css">
  <style>
    .analysis-tabs{{display:flex;flex:0 0 auto;gap:6px;padding:7px 18px;overflow-x:auto;background:var(--surface);border-bottom:1px solid var(--border)}}
    .analysis-tabs a{{flex:0 0 auto;white-space:nowrap;padding:5px 11px;border-radius:7px;color:var(--muted);font-size:13px;text-decoration:none}}
    .analysis-tabs a[aria-current=page]{{background:var(--accent-soft);color:var(--accent);font-weight:650}}
  </style>
</head>
<body class="report-shell">
  <nav class="report-bar" aria-label="Primary navigation">
    <strong>JSALT 2026 · Downsampling</strong>
    <a href="../index.html">Home</a>
    <a href="../reports/segmenter-training.html">Training report</a>
    <a href="./" aria-current="page">Analyses</a>
    <a href="../proposals/">Proposals</a>
    <a href="{SLUG}-standalone.html?v={VERSION}" target="_blank" rel="noopener">Open full page ↗</a>
  </nav>
  <nav class="analysis-tabs" aria-label="Analysis sub-tabs">
    <a href="./">All analyses</a>
    <a href="decoder-segmenter-bias.html">01 · Decoder bias</a>
    <a href="bilevel-segmenter-decoder.html">02 · Bilevel optimization</a>
    <a href="{SLUG}.html" aria-current="page">03 · Non-ASR boundaries</a>
  </nav>
  <iframe class="report-frame" src="{SLUG}-standalone.html?v={VERSION}" title="Non-ASR task results and boundary analysis"></iframe>
</body>
</html>
'''
    (research / f'{SLUG}.html').write_text(wrapper)
    index = research / 'index.html'
    content = index.read_text()
    card = '''          <a class="card card-link" href="nonasr-task-boundaries.html">
            <span class="tag green">Analysis 03</span>
            <h3>Non-ASR task adaptation and speech boundaries</h3>
            <p>54 audited task results and 2,216 utterances from CREMA-D, Expresso and LibriSpeech: token rates, shared boundaries, phone and voicing controls, quiet gaps, checkpoint selection, and proposed paper changes.</p>
            <p class="item-meta">13 September 2026 · Completed analysis · 6 figures</p>
          </a>
'''
    if f'href="{SLUG}.html"' not in content:
        assert content.count('<div class="card-grid">') == 1
        content = content.replace('<div class="card-grid">', '<div class="card-grid">\n' + card, 1)
    index.write_text(content)
    training_wrapper = args.output / 'reports/segmenter-training.html'
    content = training_wrapper.read_text()
    content = re.sub(r'segmenter-training-standalone\.html\?v=[^"\s]+',
                     f'segmenter-training-standalone.html?v={VERSION}', content)
    if f'{SLUG}.html' not in content:
        content = content.replace('<a href="../research/">Analyses</a>',
            f'<a href="../research/">Analyses</a>\n    <a href="../research/{SLUG}.html">Non-ASR boundaries</a>')
    training_wrapper.write_text(content)
    shutil.copyfile(args.training_report, args.output / 'reports/segmenter-training-standalone.html')
    changed = ['research/index.html', f'research/{SLUG}.html', f'research/{SLUG}-standalone.html',
               'reports/segmenter-training.html', 'reports/segmenter-training-standalone.html']
    changed.extend(str(p.relative_to(args.output)) for p in sorted((research / asset_dir.name).iterdir()) if p.is_file())
    (args.output.parent / 'publication_files.json').write_text(json.dumps(changed, indent=2) + '\n')
    print(json.dumps({'staged_site': str(args.output), 'publication_files': len(changed), 'version': VERSION}, indent=2))


if __name__ == '__main__':
    main()
