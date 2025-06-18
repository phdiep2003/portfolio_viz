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

// --- Toggle Portfolio Plot Tabs ---
document.addEventListener('DOMContentLoaded', () => {
  const buttons = document.querySelectorAll('.btn-group button');
  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      buttons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      
      const freq = btn.getAttribute('data-freq');
      document.querySelectorAll('.portfolio-plot').forEach(div => div.style.display = 'none');
      
      const selectedPlot = document.getElementById(`plot-${freq}`);
      if (selectedPlot) selectedPlot.style.display = 'block';
    });
  });
});
function injectPlotlyHTML(containerId, htmlString) {
  const container = document.getElementById(containerId);
  if (!container) return;

  // Create a temporary wrapper div
  const tempDiv = document.createElement('div');
  tempDiv.innerHTML = htmlString;

  // Move all children (including div + script) to the container
  container.innerHTML = '';
  while (tempDiv.firstChild) {
    container.appendChild(tempDiv.firstChild);
  }

  // Re-execute any script tags
  const scripts = container.querySelectorAll("script");
  scripts.forEach((script) => {
    const newScript = document.createElement("script");
    if (script.src) {
      newScript.src = script.src;
    } else {
      newScript.textContent = script.textContent;
    }
    document.body.appendChild(newScript);
    document.body.removeChild(newScript);
  });
}

document.addEventListener("DOMContentLoaded", function () {
  const container = document.querySelector(".results");
  if (!container) return;

  const cacheKey = container.dataset.cacheKey;
  const startDate = container.dataset.startDate;
  const endDate = container.dataset.endDate;

  function loadPlot(url, targetId) {
    fetch(url)
      .then((res) => res.json())
      .then((data) => {
        if (data.html) {
          injectPlotlyHTML(targetId, data.html);
        } else if (data.error) {
          document.getElementById(targetId).innerHTML = `<p>${data.error}</p>`;
        } else {
          document.getElementById(targetId).innerHTML = "<p>Plot data missing.</p>";
        }
      })
      .catch((err) => {
        const target = document.getElementById(targetId);
        if (target) target.innerHTML = "<p>Error loading plot.</p>";
        console.error("Plot load error:", err);
      });
  }

  if (cacheKey && startDate && endDate) {
    loadPlot(`/plot/efficient_frontier?cache_key=${cacheKey}`, "efficient-frontier-container");
    loadPlot(`/plot/nav_chart?cache_key=${cacheKey}&rebalance=monthly&start_date=${startDate}&end_date=${endDate}`, "nav-plot-monthly");
    loadPlot(`/plot/nav_chart?cache_key=${cacheKey}&rebalance=weekly&start_date=${startDate}&end_date=${endDate}`, "nav-plot-weekly");
    loadPlot(`/plot/heatmap?cache_key=${cacheKey}`, "heatmap-container");
  } else {
    console.warn("Missing cache key or date range for plots");
  }
});
