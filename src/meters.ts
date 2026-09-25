/**
 * Teachable-Machine-style probability meters, shared by both pages.
 *
 * Lives here rather than in one page's module because the shapes demo and the
 * Studio both need it — the demo previously showed only a verdict line, which
 * made the two pages look inconsistent for no reason.
 *
 * Bars are REUSED rather than re-created: replacing innerHTML every prediction
 * restarts the element, so the CSS width transition never runs and the display
 * looks frozen. Keeping the nodes and writing only `style.width` lets the browser
 * animate, which matters in continuous mode where predictions arrive every few
 * hundred ms.
 */
export function renderVerdict(
  host: HTMLElement,
  classes: string[],
  res: { predicted: string; confidence: number; probs: number[] },
  out: { matched: boolean; delivered: string; skipped?: string },
  source: string,
  threshold: number,
  targets: string[],
  /** ms between grabbing the frame and having a verdict; shown when it matters */
  staleMs?: number,
): void {
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  let head = host.querySelector('.vhead') as HTMLElement | null;
  let bars = host.querySelector('.bars') as HTMLElement | null;

  if (!bars || bars.dataset.classes !== classes.join('|')) {
    host.innerHTML =
      `<div class="vhead"></div>` +
      `<div class="bars" data-classes="${classes.join('|')}">` +
      classes.map(c =>
        `<div class="brow" data-c="${c}">` +
        `<span class="bname">${c}</span>` +
        `<i class="btrack"><b class="bfill"></b><u class="bthr"></u></i>` +
        `<em class="bval">0%</em></div>`).join('') +
      `</div>`;
    head = host.querySelector('.vhead') as HTMLElement;
    bars = host.querySelector('.bars') as HTMLElement;
  }

  head!.innerHTML =
    `<span class="${out.matched ? 'g' : 'o'}" style="font-size:18px;font-weight:700">` +
    `${out.matched ? '\u25cf MATCH' : '\u25cb no match'}</span> <b>${res.predicted}</b> ` +
    `<span class="dim">${pct(res.confidence)} \u00b7 ${source}` +
    `${staleMs !== undefined && staleMs > 250 ? ` \u00b7 <b class="o">${staleMs} ms old</b>` : ''}` +
    ` \u00b7 ${out.delivered}` +
    `${out.skipped ? ' (cooldown)' : ''}</span>`;

  classes.forEach((c, i) => {
    const row = bars!.querySelector(`.brow[data-c="${c}"]`) as HTMLElement;
    if (!row) return;
    const p = res.probs[i];
    const fill = row.querySelector('.bfill') as HTMLElement;
    const val = row.querySelector('.bval') as HTMLElement;
    const thr = row.querySelector('.bthr') as HTMLElement;
    fill.style.width = `${(p * 100).toFixed(1)}%`;
    val.textContent = `${(p * 100).toFixed(1)}%`;
    // colour encodes WHY the rule fired, not just which class is largest
    const isWinner = c === res.predicted;
    const isTarget = targets.includes(c);
    fill.className = 'bfill' + (isWinner && isTarget && p >= threshold ? ' hit'
                              : isWinner ? ' win' : '');
    row.classList.toggle('target', isTarget);
    thr.style.display = isTarget ? '' : 'none';
    thr.style.left = `${(threshold * 100).toFixed(1)}%`;
  });
}
