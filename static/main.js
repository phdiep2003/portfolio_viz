// --- jQuery UI Autocomplete Initialization ---
function initializeAutocomplete(input) {
  $(input).autocomplete({
    source: function (request, response) {
      $.ajax({
        url: "/api/tickers",
        data: { q: request.term },
        success: function (data) {
          response(data);
        },
        error: function () {
          response([]);
        }
      });
    },
    minLength: 1,
    autoFocus: true,
    open: function () {
      const menu = $(this).data('ui-autocomplete').menu.element;
      menu.css('width', $(this).outerWidth() + 'px');
    },
    select: function (event, ui) {
      $(this).val(ui.item.value);
      return false;
    }
  }).on("keydown", function (e) {
    const ac = $(this).data("ui-autocomplete");
    if ((e.key === "Tab" || e.key === "Enter") && ac && ac.menu.active) {
      e.preventDefault();
      ac.menu.select();
    }
  });
}

// --- Bind Autocomplete on Page Load ---
$(document).ready(function () {
  $('.ticker-input').each(function () {
    initializeAutocomplete(this);
  });

  // ✅ SAFE SUBMIT HANDLING HERE
  $('form').on('submit', function () {
    const btn = this.querySelector('.submit-btn');
    if (btn) {
      btn.disabled = true;
      btn.textContent = 'Running...';
    }
  });
});

// --- Dynamically Add New Asset Row ---
function addRow() {
  const tbody = document.getElementById('assets-tbody');
  const rowCount = tbody.rows.length;
  const newRow = document.createElement('tr');

  newRow.innerHTML = `
      <td>
          <input type="text" name="tickers_${rowCount}" class="ticker-input" autocomplete="off" data-index="${rowCount}">
      </td>
      <td>
          <div class="input-with-symbol">
              <input type="number" name="min_${rowCount}" min="0" max="100" step="0.01" value="0">
              <span class="percent-symbol">%</span>
          </div>
      </td>
      <td>
          <div class="input-with-symbol">
              <input type="number" name="max_${rowCount}" min="0" max="100" step="0.01" value="15">
              <span class="percent-symbol">%</span>
          </div>
      </td>
  `;

  tbody.appendChild(newRow);
  document.getElementById('asset_count').value = rowCount + 1;
  initializeAutocomplete(newRow.querySelector('.ticker-input'));
}

// --- Copy HTML Table to Clipboard ---
function copyTable(tableId) {
  const table = document.getElementById(tableId);
  if (!table) return alert("Table not found!");

  let text = '';
  for (const row of table.rows) {
    const rowText = [...row.cells].map(cell => cell.innerText.trim());
    text += rowText.join('\t') + '\n';
  }

  navigator.clipboard.writeText(text)
    .then(() => alert("Table copied to clipboard as plain text!"))
    .catch(err => alert("Failed to copy: " + err));
}

// --- Plotly JSON Plot Loader ---
function loadPlotJson(url, containerId) {
  const container = document.getElementById(containerId);
  if (!container) return Promise.resolve(); // so Promise.all doesn't fail

  return fetch(url)
    .then(res => res.json())
    .then(data => {
      console.log("Plot data received for", containerId, data);
      if (data.error) {
        container.innerHTML = `<p>${data.error}</p>`;
        return;
      }

      if (data.data && data.layout) {
        const hasPlot = container.classList.contains("js-plotly-plot");
        const plotFunc = hasPlot ? Plotly.react : Plotly.newPlot;
        return plotFunc(container, data.data, data.layout, { responsive: true });
      } else {
        container.innerHTML = "<p>Plot data missing.</p>";
      }
    })
    .catch(err => {
      container.innerHTML = "<p>Error loading plot.</p>";
      console.error("Plot load error:", err);
    });
}
// --- Toggle Portfolio Plot Tabs ---
document.addEventListener("DOMContentLoaded", async () => {
  const container = document.querySelector(".results");
  if (!container) {
    console.warn("No results container found, skipping plot loading.");
    return;
  }

  // Hide while loading
  container.style.visibility = "hidden";

  const cacheKey = container.dataset.cacheKey;
  const startDate = container.dataset.startDate;
  const endDate = container.dataset.endDate;

  if (!(cacheKey && startDate && endDate)) {
    console.warn("Missing cache key or date range for plots.");
    return;
  }

  try {
    await Promise.all([
      loadPlotJson(`/plot/efficient_frontier?cache_key=${cacheKey}`, "efficient-frontier-container"),
      loadPlotJson(`/plot/heatmap?cache_key=${cacheKey}`, "heatmap-container"),
      loadPlotJson(`/plot/nav_chart?cache_key=${cacheKey}&rebalance=monthly&start_date=${startDate}&end_date=${endDate}`, "nav-plot-monthly")
      // loadPlotJson(`/plot/nav_chart?cache_key=${cacheKey}&rebalance=weekly&start_date=${startDate}&end_date=${endDate}`, "nav-plot-weekly")
    ]);

    console.log("All plots loaded successfully");
    container.style.visibility = "visible";  // Show after all plots are ready
  } catch (err) {
    console.error("Error loading plots:", err);
    container.style.visibility = "visible"; // Still show to let error messages appear
  }

  // Setup toggle buttons
  const freqButtons = document.querySelectorAll('.btn-group button');
  freqButtons.forEach(button => {
    button.addEventListener('click', () => {
      freqButtons.forEach(b => b.classList.remove('active'));
      button.classList.add('active');

      const freq = button.getAttribute('data-freq');
      document.querySelectorAll('.portfolio-plot').forEach(div => div.style.display = 'none');
      const selectedPlot = document.getElementById(`plot-${freq}`);
      if (selectedPlot) selectedPlot.style.display = 'block';
    });
  });
});
