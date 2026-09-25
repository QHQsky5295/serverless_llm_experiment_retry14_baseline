# Figure QA

- Data: both complete historical runs, 4,000 rows/model; R2 diagnosis only.
- Entire arrival window and all 3,999 adjacent backend-start gaps included.
- One run/model is printed on the figures. No CI or repaired latency is invented.
- Both PDF pages: 248.4 × 190.8 pt = 3.45 × 2.65 inches.
- Times New Roman regular/bold embedded; PNG is 300 DPI.
- Rendered PNGs visually inspected: labels, legend, below-axis bold captions
  visible without overlap or clipping. Coincident 7B/3B curves reflect the data.
- Blue solid vs vermillion dashed remains distinguishable without relying solely
  on hue. The diagnostic does not compare Prime, so no fictitious Prime advantage.
- Log cadence axis is explicitly labelled; all observed positive gaps included.
- Initial r1 preview retained, r2 is the curated version with explicit run count
  and plain numeric log ticks. Earlier rendering dependency/font-cache failures
  did not change input data or overwrite old publication figures.
