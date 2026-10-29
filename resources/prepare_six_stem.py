#!/usr/bin/env python3
"""Adapt the verified dexxdean/htdemucs-coreml convert.py for six stems.
Run inside a clone of https://github.com/dexxdean/htdemucs-coreml .
No model weights are included or downloaded by this script.
"""
from pathlib import Path

p = Path('convert.py')
if not p.is_file():
    raise SystemExit('ERROR: convert.py not found. Run this from the htdemucs-coreml directory.')
s = p.read_text(encoding='utf-8')
if 'MODEL_NAME = "htdemucs_6s"' in s:
    raise SystemExit('Already adapted for htdemucs_6s. No changes made.')

replacements = [
    ('MODEL_NAME = "htdemucs"', 'MODEL_NAME = "htdemucs_6s"'),
    ('NUM_SOURCES = 4', 'NUM_SOURCES = 6'),
    ('# Demucs internal source order: drums(0), bass(1), other(2), vocals(3)\n# We reorder to vocals, drums, bass, other (typical UI / DJ convention).\nSOURCE_REORDER = [3, 0, 1, 2]\nSOURCE_NAMES = ["vocals", "drums", "bass", "other"]',
     '# Demucs 6s internal source order: drums, bass, other, vocals, piano, guitar.\n# Output order: drums, bass, vocals, guitar, piano, other.\nSOURCE_REORDER = [0, 1, 3, 5, 4, 2]\nSOURCE_NAMES = ["drums", "bass", "vocals", "guitar", "piano", "other"]'),
    ('Output: (1, 4, 2, segment_samples)  -- [vocals, drums, bass, other]',
     'Output: (1, 6, 2, segment_samples)  -- [drums, bass, vocals, guitar, piano, other]'),
    ('# Reorder sources: drums,bass,other,vocals -> vocals,drums,bass,other.',
     '# Reorder six sources to drums,bass,vocals,guitar,piano,other.'),
    ('f"Separated stems. Shape (1, 4, 2, {segment_samples}). "',
     'f"Separated stems. Shape (1, {NUM_SOURCES}, 2, {segment_samples}). "'),
    ('"MIT. Original Demucs: Copyright (c) Meta Platforms, Inc. and "\n        "affiliates, MIT License. See LICENSE and ATTRIBUTION."',
     '"Demucs source code: MIT. Pretrained model weights are not covered "\n        "by the MIT license; see https://github.com/facebookresearch/demucs/issues/327."'),
]
for old, new in replacements:
    n = s.count(old)
    if n != 1:
        raise SystemExit(f'ERROR: expected exactly one occurrence of {old[:85]!r}, found {n}. No changes made. Check converter revision.')
    s = s.replace(old, new, 1)
if 'SOURCE_REORDER = [0, 1, 3, 5, 4, 2]' not in s or 'NUM_SOURCES = 6' not in s:
    raise SystemExit('ERROR: validation failed.')
backup = Path('convert.py.original-4stem')
if backup.exists():
    raise SystemExit('ERROR: backup already exists, no changes made.')
backup.write_bytes(p.read_bytes())
p.write_text(s, encoding='utf-8')
print('SUCCESS: convert.py adapted for six stems.')
print('Backup:', backup)
print('Model: htdemucs_6s')
print('Output: drums, bass, vocals, guitar, piano, other')
