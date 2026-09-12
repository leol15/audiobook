# quanqiuguiyishidai scraper

Pulls chapter text from `quanben.io` for use as a book source in `books/<slug>/source.txt`.
Chapters are numbered sequentially at `/n/quanqiuguiyishidai/<n>.html`; the
script walks forward following each page's `rel="next"` link and stops at the
last chapter or a missing page.

```bash
python3 scrape.py                    # scrape whole book into output/, resumable
python3 scrape.py --start 5 --end 10 # a specific range
python3 scrape.py --delay 2.0        # slower, gentler on the server
```

Output: one `output/NNNN_<title>.txt` file per chapter (title line + blank
line + paragraphs). Already-downloaded chapters are skipped on rerun, so an
interrupted scrape just resumes.

To concatenate into a single source file for the `ab` pipeline:

```bash
cat output/*.txt > combined.txt
```

`output/` is gitignored — the novel text is copyrighted, same as
`books/small-chinese/source.txt`; never commit it.
