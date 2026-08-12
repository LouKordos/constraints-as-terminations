# Documentation asset inventory

This page separates manuscript-derived figures that are ready to keep from the demonstration GIFs that still need final footage. The three GIFs in the README are intentionally static and visibly labelled; they should not be mistaken for experimental evidence.

## Ready to use

| Repository asset | Manuscript source | Use |
| --- | --- | --- |
| `assets/locomposition-overview.png` | `figures/main_figure.pdf` | README hero and method overview |
| `assets/cot-and-contact-patterns.png` | `figures/cot_patterns_combined.pdf` | Main efficiency/contact-pattern result |
| `assets/sim2real-overview.png` | `figures/sim2real_compressed.pdf` | Hardware and mapping overview |
| `assets/terrain-contact-adaptation.png` | `Rebuttal/Figures/gait_comparison_terrain.pdf` | Supplemental terrain-conditioned contact timing |
| `assets/swing-height-adaptation.png` | `figures/plot_step_height_box_comparison.pdf` | Supplemental swing-height analysis |

The PNGs were exported at publication resolution and checked after conversion. If the manuscript figures change, re-export these files from the final PDFs rather than editing the raster copies.

## Replace before the public release

| Placeholder | Requested clip | Export target |
| --- | --- | --- |
| `assets/demos/go2-hardware.gif` | One continuous Go2 hardware obstacle sequence that shows the LiDAR-equipped robot approaching and clearing terrain | 16:9, 960×540 or 1280×720, 6–10 s, seamless or unobtrusive loop, stable crop, under roughly 10 MB |
| `assets/demos/anymal-c.gif` | ANYmal C traversing representative rough terrain in simulation; keep enough of the terrain visible to make the task legible | 16:9, 960×540 or 1280×720, 6–8 s, looped, no training/debug UI if possible |
| `assets/demos/spot.gif` | Spot traversing the same kind of rough-terrain distribution in simulation | 16:9, 960×540 or 1280×720, 6–8 s, looped, framing comparable to the ANYmal C clip |

Keep the filenames when replacing the placeholders so the README does not need another edit. Prefer a true-color GIF or an optimized WebP only if GitHub rendering and file size have both been checked.

## Links

- Project video: `https://youtu.be/byAA07ge4O0` (already linked from the README).
- The final personal blog-post URL, if the README should link to it. It is deliberately omitted for now rather than guessing a route.
- The final public GitHub URL after the repository is renamed, so the setup script and `CITATION.cff` can be checked against the live repository.

## Final visual check

After replacing the GIFs, view the README at desktop and narrow widths. Confirm that text remains readable, loops do not flash, robots are not cropped at either end of the motion, and the three clips communicate different embodiments rather than three near-identical simulator views.
