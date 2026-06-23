#!/usr/bin/env python3
"""
Interactive Light Curve Plotter for TESS data
==============================================

Generates interactive Plotly charts (auto-opening in the browser) for TESS
light curves. Replaces the static matplotlib PNG plots with a Desmos-style
interactive chart:

  - Scroll-to-zoom, drag-to-pan, double-click to reset
  - Hover tooltips showing exact time, flux, quality flag
  - Toggle buttons for PDC flux, SAP flux, baseline, transit highlights
  - Crosshair spikes from cursor to both axes
  - Auto-fitted y-axis to data range
  - Cream/journal theme matching the HTML viewer
  - Two-finger trackpad gestures (pan) and pinch (zoom)
  - Searchable, sortable, paginated data table
  - Optional Gaussian Process regression for noise reduction

Usage:
    from interactive_plot import plot_lightcurve_interactive

    plot_lightcurve_interactive(
        time=lc['TIME'],
        pdcsap_flux=lc['PDCSAP_FLUX'],
        sap_flux=lc.get('SAP_FLUX'),
        quality=lc.get('SAP_QUALITY'),
        tic_id='278660115',
        sector='1',
        output_html='tic_278660115_sector_1.html',  # optional
        auto_open=True,
        denoise_method=None,  # None | 'gp'
    )
"""

import os
import numpy as np

try:
    import plotly.graph_objects as go
except ImportError:
    raise ImportError(
        "Plotly is required for interactive plots.\n"
        "Install it with:  pip install plotly"
    )


# ============================================================
# DENOISING FUNCTIONS
# ============================================================

def denoise_gp(time, flux, length_scale=None, n_restarts=2, max_points=2000):
    """
    Gaussian Process regression for noise reduction.

    Fits a GP with an RBF + WhiteKernel to the data, then returns the
    predicted trend (smoothed flux). The RBF kernel captures stellar
    variability and transits; WhiteKernel captures measurement noise.

    For large datasets (> max_points), a random subset is used for fitting
    and the trend is predicted at all points. This avoids the O(n^3)
    scaling of exact GP inference.

    Parameters
    ----------
    time : array-like
        Time values (BJD - 2457000)
    flux : array-like
        Flux values (will be normalized to median=1 internally)
    length_scale : float, optional
        RBF length scale in days. If None, auto-estimated from data.
    n_restarts : int
        Number of optimizer restarts for hyperparameter tuning.
    max_points : int
        Maximum number of points to use for GP fitting (subsampling).

    Returns
    -------
    smoothed : np.ndarray
        GP-predicted smooth flux (same length as input)
    kernel : str
        String representation of the learned kernel
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    # Keep only finite points
    mask = np.isfinite(time) & np.isfinite(flux)
    t = time[mask]
    f = flux[mask]

    if len(t) < 10:
        return flux, "insufficient data"

    # Auto-estimate length scale if not provided
    if length_scale is None:
        baseline = t.max() - t.min()
        length_scale = max(0.01, baseline * 0.05)

    # Normalize to median=1 for numerical stability
    median = np.median(f)
    if median == 0 or not np.isfinite(median):
        median = 1.0
    f_norm = f / median

    # Subsample for fitting if dataset is too large
    n = len(t)
    if n > max_points:
        print(f"    Subsampling {max_points}/{n} points for GP fitting (O(n^3) scaling)...")
        rng = np.random.RandomState(42)
        fit_idx = rng.choice(n, size=max_points, replace=False)
        fit_idx.sort()  # keep time order
        t_fit = t[fit_idx]
        f_fit = f_norm[fit_idx]
    else:
        t_fit = t
        f_fit = f_norm

    # Build kernel: Constant * RBF (signal) + WhiteKernel (noise)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * \
             RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2)) + \
             WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e0))

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts, normalize_y=True)

    # Fit on the (possibly subsampled) data
    t_fit_2d = t_fit.reshape(-1, 1)
    gp.fit(t_fit_2d, f_fit)

    # Predict at ALL original points (not just the fit subset)
    t_all_2d = t.reshape(-1, 1)
    trend_norm = gp.predict(t_all_2d)
    trend = trend_norm * median

    # Build output array (NaN where input was NaN)
    result = np.full_like(flux, np.nan, dtype=float)
    result[mask] = trend
    return result, gp.kernel_


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _normalize_flux(flux):
    """Normalize flux to median = 100 (percent). Returns (normalized, median)."""
    finite = flux[np.isfinite(flux)]
    if len(finite) == 0:
        return flux, 1.0
    median = np.median(finite)
    if median == 0 or not np.isfinite(median):
        median = 1.0
    return (flux / median) * 100.0, median


def _compute_y_range(*flux_arrays):
    """Compute a tight y-axis range from multiple flux arrays (all in %)."""
    all_vals = []
    for arr in flux_arrays:
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=float).tolist()
        for v in arr:
            if np.isfinite(v):
                all_vals.append(v)

    if not all_vals:
        return 99.0, 101.0

    y_min = min(all_vals)
    y_max = max(all_vals)
    span = y_max - y_min
    if span <= 0:
        span = 1.0
    pad = span * 0.05
    y_min = min(y_min - pad, 99.5)
    y_max = max(y_max + pad, 100.5)
    return y_min, y_max


def _inject_trackpad_gestures(html_path):
    """
    Inject custom JavaScript into a Plotly HTML file for trackpad gestures.

    On Mac trackpads:
      - Two-finger scroll (wheel event, ctrlKey=false) -> PAN the chart
      - Pinch to zoom (wheel event, ctrlKey=true) -> ZOOM in/out
      - Single-click drag -> box zoom (handled by Plotly's dragMode='zoom')
      - Double-click -> reset view (handled by Plotly)
    """
    custom_js = """
<script>
(function() {
    function findPlotlyDiv() {
        var divs = document.querySelectorAll('.plotly-graph-div, .js-plotly-plot');
        return divs.length > 0 ? divs[0] : null;
    }

    function setupTrackpad() {
        var gd = findPlotlyDiv();
        if (!gd) {
            setTimeout(setupTrackpad, 100);
            return;
        }

        gd.addEventListener('wheel', function(e) {
            e.preventDefault();
            e.stopPropagation();

            var xRange = gd.layout.xaxis.range;
            var yRange = gd.layout.yaxis.range;
            if (!xRange || !yRange) return;

            var xSpan = xRange[1] - xRange[0];
            var ySpan = yRange[1] - yRange[0];

            if (e.ctrlKey || e.metaKey) {
                var factor = Math.exp(e.deltaY * 0.005);
                var newYSpan = ySpan * factor;
                var newXSpan = xSpan * factor;

                var rect = gd.getBoundingClientRect();
                var cursorX = (e.clientX - rect.left) / rect.width;
                var cursorY = 1 - (e.clientY - rect.top) / rect.height;

                var xMid = xRange[0] + xSpan * cursorX;
                var yMid = yRange[0] + ySpan * cursorY;

                Plotly.relayout(gd, {
                    'xaxis.range': [xMid - newXSpan * cursorX, xMid + newXSpan * (1 - cursorX)],
                    'yaxis.range': [yMid - newYSpan * cursorY, yMid + newYSpan * (1 - cursorY)]
                });
            } else {
                var xPan = (e.deltaX / gd.clientWidth) * xSpan;
                var yPan = (e.deltaY / gd.clientHeight) * ySpan;

                Plotly.relayout(gd, {
                    'xaxis.range': [xRange[0] - xPan, xRange[1] - xPan],
                    'yaxis.range': [yRange[0] + yPan, yRange[1] + yPan]
                });
            }
        }, { passive: false });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupTrackpad);
    } else {
        setupTrackpad();
    }
})();
</script>
"""

    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()

    if '</body>' in html:
        html = html.replace('</body>', custom_js + '\n</body>')
    else:
        html += custom_js

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)


def _inject_data_table(html_path, data, tic_id, sector, denoise_info=None):
    """
    Inject a searchable, paginated data table below the Plotly chart.
    """
    import json as _json

    data_json = _json.dumps(data)

    denoise_banner = ""
    if denoise_info:
        denoise_banner = '<div style="background: #ebe5d4; border: 1px solid #d6cdb5; border-radius: 4px; padding: 8px 12px; margin-bottom: 12px; font-family: \'JetBrains Mono\', monospace; font-size: 11px; color: #2c2a26;">' + denoise_info + '</div>'

    html_block = """
<div id="lc-table-section" style="margin-top: 20px; display: none;">
  <div style="background: #faf7f0; border: 1px solid #d6cdb5; border-radius: 8px; padding: 16px;">
    """ + denoise_banner + """
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap;">
      <h3 style="font-family: Lora, Georgia, serif; font-size: 16px; color: #2c2a26; margin: 0;">
        Light Curve Data
      </h3>
      <span id="lc-row-count" style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #6b6354;"></span>
      <input type="text" id="lc-search" placeholder="Search (time, flux, quality)..."
             style="flex: 1; min-width: 200px; padding: 5px 10px; border: 1px solid #d6cdb5;
                    border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 12px;
                    background: #faf7f0; color: #2c2a26;">
      <button id="lc-export-csv" style="padding: 5px 12px; background: #b8651a; color: white;
              border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">
        Export CSV
      </button>
    </div>
    <div style="overflow-x: auto;">
      <table id="lc-table" style="width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 11px;">
        <thead>
          <tr style="background: #f3eee2; border-bottom: 2px solid #d6cdb5;">
            <th class="lc-sortable" data-col="index" style="padding: 6px 10px; text-align: right; cursor: pointer; color: #2c2a26;">#</th>
            <th class="lc-sortable" data-col="time" style="padding: 6px 10px; text-align: right; cursor: pointer; color: #2c2a26;">Time (BJD-2457000)</th>
            <th class="lc-sortable" data-col="pdc_pct" style="padding: 6px 10px; text-align: right; cursor: pointer; color: #2c2a26;">PDC Flux (%)</th>
            <th class="lc-sortable" data-col="pdc_raw" style="padding: 6px 10px; text-align: right; cursor: pointer; color: #2c2a26;">PDC Flux (e-/s)</th>
            <th class="lc-sortable" data-col="sap_pct" style="padding: 6px 10px; text-align: right; cursor: pointer; color: #2c2a26; display: none;">SAP Flux (%)</th>
            <th class="lc-sortable" data-col="quality" style="padding: 6px 10px; text-align: right; cursor: pointer; color: #2c2a26;">Quality</th>
          </tr>
        </thead>
        <tbody id="lc-table-body"></tbody>
      </table>
    </div>
    <div id="lc-pagination" style="display: flex; align-items: center; gap: 8px; margin-top: 12px; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #6b6354;">
      <button id="lc-prev-page" style="padding: 4px 10px; background: #ebe5d4; border: 1px solid #d6cdb5; border-radius: 3px; cursor: pointer;">Prev</button>
      <span id="lc-page-info"></span>
      <button id="lc-next-page" style="padding: 4px 10px; background: #ebe5d4; border: 1px solid #d6cdb5; border-radius: 3px; cursor: pointer;">Next</button>
    </div>
  </div>
</div>

<style>
  .lc-sortable:hover { background: #ebe5d4 !important; }
  .lc-sortable.sorted-asc::after { content: " \\25B2"; font-size: 9px; color: #b8651a; }
  .lc-sortable.sorted-desc::after { content: " \\25BC"; font-size: 9px; color: #b8651a; }
  #lc-table tbody tr:hover { background: #f3eee2; }
  #lc-table tbody tr.bad-quality { color: #c9444a; }
  #lc-table td { padding: 3px 10px; border-bottom: 1px solid #ebe5d4; }
</style>

<script>
(function() {
    var DATA = """ + data_json + """;
    var HAS_SAP = DATA.sap_flux_pct !== undefined;
    var ROWS_PER_PAGE = 50;
    var currentPage = 0;
    var filteredRows = [];
    var sortCol = 'index';
    var sortDir = 'asc';

    function buildRows() {
        var rows = [];
        for (var i = 0; i < DATA.time.length; i++) {
            rows.push({
                index: i,
                time: DATA.time[i],
                pdc_pct: DATA.pdcsap_flux_pct[i],
                pdc_raw: DATA.pdcsap_flux_raw[i],
                sap_pct: HAS_SAP ? DATA.sap_flux_pct[i] : null,
                quality: DATA.quality[i]
            });
        }
        return rows;
    }

    var allRows = buildRows();
    filteredRows = allRows.slice();

    if (HAS_SAP) {
        var sapTh = document.querySelector('[data-col="sap_pct"]');
        if (sapTh) sapTh.style.display = '';
    }

    function filterRows(query) {
        if (!query) {
            filteredRows = allRows.slice();
        } else {
            var q = query.toLowerCase();
            filteredRows = allRows.filter(function(r) {
                return String(r.time).indexOf(q) >= 0 ||
                       String(r.pdc_pct).indexOf(q) >= 0 ||
                       String(r.pdc_raw).indexOf(q) >= 0 ||
                       (r.sap_pct !== null && String(r.sap_pct).indexOf(q) >= 0) ||
                       String(r.quality).indexOf(q) >= 0 ||
                       String(r.index).indexOf(q) >= 0;
            });
        }
        currentPage = 0;
        sortRows();
        renderTable();
    }

    function sortRows() {
        filteredRows.sort(function(a, b) {
            var va = a[sortCol], vb = b[sortCol];
            if (va === null) return 1;
            if (vb === null) return -1;
            if (typeof va === 'number' && typeof vb === 'number') {
                return sortDir === 'asc' ? va - vb : vb - va;
            }
            return sortDir === 'asc'
                ? String(va).localeCompare(String(vb))
                : String(vb).localeCompare(String(va));
        });
    }

    function renderTable() {
        var tbody = document.getElementById('lc-table-body');
        var start = currentPage * ROWS_PER_PAGE;
        var end = Math.min(start + ROWS_PER_PAGE, filteredRows.length);
        var html = '';
        for (var i = start; i < end; i++) {
            var r = filteredRows[i];
            var badClass = r.quality !== 0 ? ' bad-quality' : '';
            html += '<tr class="' + badClass + '">';
            html += '<td style="text-align:right;">' + r.index + '</td>';
            html += '<td style="text-align:right;">' + r.time.toFixed(5) + '</td>';
            html += '<td style="text-align:right;">' + (isFinite(r.pdc_pct) ? r.pdc_pct.toFixed(4) : 'NaN') + '</td>';
            html += '<td style="text-align:right;">' + (isFinite(r.pdc_raw) ? r.pdc_raw.toFixed(2) : 'NaN') + '</td>';
            if (HAS_SAP) {
                html += '<td style="text-align:right;">' + (r.sap_pct !== null && isFinite(r.sap_pct) ? r.sap_pct.toFixed(4) : 'NaN') + '</td>';
            }
            html += '<td style="text-align:right;">' + r.quality + '</td>';
            html += '</tr>';
        }
        tbody.innerHTML = html;

        document.getElementById('lc-row-count').textContent =
            filteredRows.length + ' rows' + (filteredRows.length !== allRows.length ? ' (of ' + allRows.length + ')' : '');
        var totalPages = Math.ceil(filteredRows.length / ROWS_PER_PAGE);
        document.getElementById('lc-page-info').textContent =
            'Page ' + (currentPage + 1) + ' of ' + Math.max(totalPages, 1);
        document.getElementById('lc-prev-page').disabled = currentPage === 0;
        document.getElementById('lc-next-page').disabled = currentPage >= totalPages - 1;

        document.querySelectorAll('.lc-sortable').forEach(function(th) {
            th.classList.remove('sorted-asc', 'sorted-desc');
            if (th.dataset.col === sortCol) {
                th.classList.add(sortDir === 'asc' ? 'sorted-asc' : 'sorted-desc');
            }
        });
    }

    document.getElementById('lc-search').addEventListener('input', function(e) {
        filterRows(e.target.value.trim());
    });

    document.querySelectorAll('.lc-sortable').forEach(function(th) {
        th.addEventListener('click', function() {
            var col = th.dataset.col;
            if (sortCol === col) {
                sortDir = sortDir === 'asc' ? 'desc' : 'asc';
            } else {
                sortCol = col;
                sortDir = 'asc';
            }
            sortRows();
            renderTable();
        });
    });

    document.getElementById('lc-prev-page').addEventListener('click', function() {
        if (currentPage > 0) { currentPage--; renderTable(); }
    });
    document.getElementById('lc-next-page').addEventListener('click', function() {
        var totalPages = Math.ceil(filteredRows.length / ROWS_PER_PAGE);
        if (currentPage < totalPages - 1) { currentPage++; renderTable(); }
    });

    document.getElementById('lc-export-csv').addEventListener('click', function() {
        var csv = 'index,time_bjd,pdc_flux_pct,pdc_flux_raw';
        if (HAS_SAP) csv += ',sap_flux_pct';
        csv += ',quality\\n';
        for (var i = 0; i < filteredRows.length; i++) {
            var r = filteredRows[i];
            csv += r.index + ',' + r.time.toFixed(6) + ',' + r.pdc_pct + ',' + r.pdc_raw;
            if (HAS_SAP) csv += ',' + (r.sap_pct !== null ? r.sap_pct : '');
            csv += ',' + r.quality + '\\n';
        }
        var blob = new Blob([csv], { type: 'text/csv' });
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = 'tic_""" + str(tic_id) + """_sector_""" + str(sector) + """_data.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });

    renderTable();
})();
</script>
"""

    toggle_button = """
<style>
#lc-table-toggle {
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 9999;
    padding: 10px 22px;
    background: #b8651a;
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    font-family: Lora, Georgia, serif;
    font-weight: 500;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    transition: background 0.15s;
}
#lc-table-toggle:hover { background: #9c5414; }
</style>
<button id="lc-table-toggle">Show Data Table</button>
<script>
document.getElementById('lc-table-toggle').addEventListener('click', function() {
    var section = document.getElementById('lc-table-section');
    if (section.style.display === 'none') {
        section.style.display = 'block';
        this.textContent = 'Hide Data Table';
        setTimeout(function() {
            section.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 50);
    } else {
        section.style.display = 'none';
        this.textContent = 'Show Data Table';
    }
});
</script>
"""

    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()

    injection = toggle_button + '\n' + html_block
    if '</body>' in html:
        html = html.replace('</body>', injection + '\n</body>')
    else:
        html += injection

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)


# ============================================================
# MAIN PLOTTING FUNCTION
# ============================================================

def plot_lightcurve_interactive(
    time,
    pdcsap_flux,
    sap_flux=None,
    quality=None,
    tic_id='unknown',
    sector='?',
    output_html=None,
    auto_open=True,
    show_baseline=True,
    show_transits=True,
    title=None,
    denoise_method=None,
    denoise_gp_length_scale=None,
):
    """
    Generate an interactive Plotly light curve chart and open it in the browser.

    Parameters
    ----------
    time : array-like
        TIME column from the FITS file (BJD - 2457000, days)
    pdcsap_flux : array-like
        PDCSAP_FLUX column (cleaned brightness)
    sap_flux : array-like, optional
        SAP_FLUX column (raw brightness).
    quality : array-like, optional
        SAP_QUALITY column.
    tic_id : str
        TIC ID of the star (for the chart title)
    sector : str
        TESS sector number (for the chart title)
    output_html : str, optional
        Path to save the interactive HTML file.
    auto_open : bool
        If True, automatically open the chart in the default browser.
    show_baseline : bool
        If True, draw a red dashed line at 100% (the median).
    show_transits : bool
        If True, highlight transit candidates using rolling-median + MAD.
    title : str, optional
        Custom chart title.
    denoise_method : str, optional
        None (default): no denoising, plot raw PDCSAP_FLUX
        'gp': apply Gaussian Process regression (RBF + WhiteKernel)
    denoise_gp_length_scale : float, optional
        RBF length scale in days for GP (default: auto-estimated).

    Returns
    -------
    str
        Path to the saved HTML file.
    """
    time = np.asarray(time, dtype=float)
    pdcsap_flux = np.asarray(pdcsap_flux, dtype=float)

    # ---- Apply denoising if requested ----
    denoise_info = None
    plot_flux = pdcsap_flux.copy()  # default: no denoising

    if denoise_method == 'gp':
        print("  Applying Gaussian Process regression...")
        try:
            smoothed, kernel = denoise_gp(
                time, pdcsap_flux,
                length_scale=denoise_gp_length_scale
            )
            plot_flux = smoothed
            denoise_info = (
                "Denoised with Gaussian Process Regression | "
                "Kernel: " + str(kernel)[:80] + "..."
            )
            print("    GP kernel:", kernel)
        except Exception as e:
            print(f"  GP denoising failed: {e}, using raw flux")
            denoise_info = "GP denoising failed - showing raw flux"

    # ---- Normalize PDC flux to % ----
    pdc_norm, pdc_median = _normalize_flux(plot_flux)

    # ---- Normalize SAP flux (if provided) to the SAME median ----
    sap_norm = None
    if sap_flux is not None:
        sap_flux = np.asarray(sap_flux, dtype=float)
        sap_norm = (sap_flux / pdc_median) * 100.0

    # ---- Filter non-finite points for plotting ----
    if quality is not None:
        quality = np.asarray(quality, dtype=int)
    pdc_mask = np.isfinite(time) & np.isfinite(pdc_norm)
    t_pdc = np.asarray(time[pdc_mask], dtype=float).tolist()
    f_pdc = np.asarray(pdc_norm[pdc_mask], dtype=float).tolist()
    q_pdc = quality[pdc_mask].tolist() if quality is not None else np.zeros(len(t_pdc), dtype=int).tolist()

    # ============================================================
    # BUILD TRACES
    # ============================================================
    traces = []

    # PDC flux trace (terracotta, main)
    traces.append(go.Scattergl(
        x=t_pdc,
        y=f_pdc,
        mode='markers',
        name='PDC flux' + (f' ({denoise_method})' if denoise_method else ''),
        marker=dict(size=3, color='#b8651a', opacity=0.6),
        customdata=q_pdc,
        hovertemplate=(
            '<b>Time:</b> %{x:.5f} days<br>'
            '<b>Flux:</b> %{y:.4f}%<br>'
            '<b>Quality:</b> %{customdata}'
            '<extra></extra>'
        ),
    ))

    # SAP flux trace (sage green, toggleable)
    if sap_norm is not None:
        sap_mask = np.isfinite(time) & np.isfinite(sap_norm)
        traces.append(go.Scattergl(
            x=np.asarray(time[sap_mask], dtype=float).tolist(),
            y=np.asarray(sap_norm[sap_mask], dtype=float).tolist(),
            mode='markers',
            name='SAP flux (raw)',
            marker=dict(size=3, color='#5a7a3a', opacity=0.5),
            visible=False,
            hovertemplate=(
                '<b>Time:</b> %{x:.5f} days<br>'
                '<b>SAP Flux:</b> %{y:.4f}%<br>'
                '<extra></extra>'
            ),
        ))

    # Transit highlights - rolling median + MAD detection
    if show_transits and len(f_pdc) > 50:
        f_pdc_arr = np.array(f_pdc)
        t_arr = np.array(t_pdc)

        window = max(20, len(f_pdc_arr) // 20)
        try:
            from scipy.ndimage import median_filter
            baseline = median_filter(f_pdc_arr, size=window, mode='nearest')
        except ImportError:
            baseline = np.array(f_pdc_arr, dtype=float)
            half_w = window // 2
            for i in range(len(f_pdc_arr)):
                lo = max(0, i - half_w)
                hi = min(len(f_pdc_arr), i + half_w + 1)
                baseline[i] = np.median(f_pdc_arr[lo:hi])

        residuals = f_pdc_arr - baseline
        mad = np.median(np.abs(residuals - np.median(residuals)))
        sigma = 1.4826 * mad

        if sigma > 0:
            transit_mask = residuals < (-3 * sigma)
            transit_mask = transit_mask & (f_pdc_arr < baseline - 0.1)

            if np.any(transit_mask):
                traces.append(go.Scattergl(
                    x=t_arr[transit_mask].tolist(),
                    y=f_pdc_arr[transit_mask].tolist(),
                    mode='markers',
                    name='Transit candidates',
                    marker=dict(size=6, color='#d49422', opacity=0.9,
                                line=dict(color='#d49422', width=1)),
                    hovertemplate=(
                        '<b>Transit candidate</b><br>'
                        '<b>Time:</b> %{x:.5f} days<br>'
                        '<b>Flux:</b> %{y:.4f}%<br>'
                        '<b>Depth:</b> %{customdata:.3f}%<br>'
                        '<extra></extra>'
                    ),
                    customdata=(baseline[transit_mask] - f_pdc_arr[transit_mask]).tolist(),
                ))

    # ============================================================
    # Y-AXIS RANGE
    # ============================================================
    sap_for_range = None
    if sap_norm is not None:
        sap_for_range = np.asarray(sap_norm[sap_mask], dtype=float).tolist()
    y_min, y_max = _compute_y_range(f_pdc, sap_for_range)

    # ============================================================
    # BASELINE SHAPE
    # ============================================================
    shapes = []
    if show_baseline:
        shapes.append(dict(
            type='line',
            xref='paper', x0=0, x1=1,
            yref='y', y0=100, y1=100,
            line=dict(color='#c9444a', width=1.5, dash='dash'),
        ))

    # ============================================================
    # LAYOUT
    # ============================================================
    chart_title = title or f'TIC {tic_id} - Sector {sector}'
    if denoise_method == 'gp':
        chart_title += ' <span style="font-size:12px;color:#5B6B7D">(GP-denoised)</span>'

    layout = go.Layout(
        title=dict(
            text=chart_title,
            font=dict(family='Lora, Georgia, serif', size=16, color='#2c2a26'),
            x=0.5,
        ),
        paper_bgcolor='#faf7f0',
        plot_bgcolor='#faf7f0',
        font=dict(family='Lora, Georgia, serif', size=12, color='#6b6354'),
        margin=dict(l=70, r=24, t=60, b=50),
        xaxis=dict(
            title=dict(text='Time (BJD - 2457000, days)', font=dict(size=12)),
            gridcolor='#d6cdb5',
            zerolinecolor='#d6cdb5',
            linecolor='#d6cdb5',
            showspikes=True,
            spikethickness=1,
            spikecolor='#b8651a',
            spikemode='toaxis+across+marker',
            spikesnap='data',
        ),
        yaxis=dict(
            title=dict(text='Normalized Flux (%)', font=dict(size=12)),
            gridcolor='#d6cdb5',
            zerolinecolor='#d6cdb5',
            linecolor='#d6cdb5',
            range=[y_min, y_max],
            showspikes=True,
            spikethickness=1,
            spikecolor='#b8651a',
            spikemode='toaxis+across+marker',
            spikesnap='data',
        ),
        hovermode='closest',
        hoverdistance=30,
        dragmode='zoom',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=11),
            bgcolor='rgba(243, 238, 226, 0.8)',
        ),
        shapes=shapes,
    )

    fig = go.Figure(data=traces, layout=layout)

    config = {
        'responsive': True,
        'displayModeBar': True,
        'scrollZoom': False,
        'doubleClick': 'reset',
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'tic_{tic_id}_sector_{sector}',
            'height': 500,
            'width': 1200,
            'scale': 2,
        },
    }

    # ============================================================
    # SAVE
    # ============================================================
    if output_html is None:
        output_html = f'tic_{tic_id}_sector_{sector}_interactive.html'
    output_html = os.path.abspath(output_html)

    fig.write_html(output_html, config=config, include_plotlyjs='cdn')

    # Inject trackpad gestures
    _inject_trackpad_gestures(output_html)

    # Inject data table
    import json as _json
    table_data = {
        'time': t_pdc,
        'pdcsap_flux_pct': f_pdc,
        'pdcsap_flux_raw': (np.array(f_pdc) * pdc_median / 100.0).tolist(),
        'quality': q_pdc,
    }
    if sap_norm is not None:
        table_data['sap_flux_pct'] = np.asarray(sap_norm[sap_mask], dtype=float).tolist()
    _inject_data_table(output_html, table_data, tic_id, sector, denoise_info)

    print(f"\nInteractive chart saved to: {output_html}")

    # Print stats
    n_finite = len(t_pdc)
    n_transits = 0
    if show_transits:
        for tr in traces:
            if tr.name == 'Transit candidates':
                n_transits = len(tr.x)
                break
    time_span = float(max(t_pdc) - min(t_pdc)) if n_finite > 1 else 0
    print(f"  Cadences plotted: {n_finite:,}")
    print(f"  Time span: {time_span:.2f} days")
    print(f"  Flux range: {min(f_pdc):.4f}% - {max(f_pdc):.4f}%")
    print(f"  Transit candidates (3-sigma): {n_transits:,}")
    print(f"  Median flux: {pdc_median:.2f} e-/s")
    if denoise_info:
        print(f"  Denoise: {denoise_info}")

    if auto_open:
        import webbrowser
        print(f"  Opening in browser...")
        webbrowser.open('file://' + output_html)

    return output_html


if __name__ == '__main__':
    # Demo with synthetic data + both denoising methods
    print("Generating demo light curve with noise...")
    np.random.seed(42)
    n = 2000
    t = 1325 + np.arange(n) * 0.014
    # Add stellar variability (sinusoidal) + noise + transits
    base_flux = 355400 + np.sin(np.arange(n) * 0.05) * 500 + np.random.normal(0, 150, n)
    for start in [400, 1100, 1700]:
        base_flux[start:start+6] *= 0.97  # 3% transit dips

    quality = np.zeros(n, dtype=int)

    # Test 1: No denoising
    print("\n=== Test 1: Raw (no denoising) ===")
    plot_lightcurve_interactive(
        time=t, pdcsap_flux=base_flux, quality=quality,
        tic_id='DEMO_RAW', sector='1',
        output_html='demo_raw.html',
        auto_open=False,
        denoise_method=None,
    )

    # Test 2: GP denoising
    print("\n=== Test 2: Gaussian Process denoising ===")
    plot_lightcurve_interactive(
        time=t, pdcsap_flux=base_flux, quality=quality,
        tic_id='DEMO_GP', sector='1',
        output_html='demo_gp.html',
        auto_open=False,
        denoise_method='gp',
    )

    print("\nBoth demos generated. Open them in a browser to compare.")
