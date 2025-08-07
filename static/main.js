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
    .then(() => alert("Table copied to clipboard!"))
    .catch(err => alert("Failed to copy: " + err));
}

// --- Function to render all results from API data ---
function renderResults(data) {
  // 1. Define the HTML structure for all results sections.
  const resultsHtml = `
      <div class="results-section">
        <h2>Efficient Frontier</h2>
        <div id="efficient-frontier-container"></div>
      </div>
      <div class="results-section">
        <h2>Key Metrics</h2>
        <button class="copy-btn" onclick="copyTable('mu_table')">Copy 📄</button>
        <div id="mu-table-container" class="table-responsive"></div>
      </div>
      <div class="results-section">
        <h2>Optimized Performance</h2>
        <button class="copy-btn" onclick="copyTable('perf-summary')">Copy 📄</button>
        <div id="perf-summary-container" class="table-responsive"></div>
      </div>
      <div class="results-section">
        <h2>Asset Allocation Analysis</h2>
        <div id="heatmap-container"></div>
        <div id="allocation-table-container" class="table-responsive"></div>
      </div>
      <div class="results-section">
        <h2>Correlation Matrix</h2>
        <button class="copy-btn" onclick="copyTable('correlation-matrix')">Copy 📄</button>
        <div id="correlation-matrix-container" class="table-responsive"></div>
      </div>
      <div class="results-section">
        <h2>Portfolio NAV Plots</h2>
        <div id="nav-plot-monthly"></div>
      </div>
  `;
  $('#results-container').hide().html(resultsHtml).fadeIn(400);
  $('html, body').animate({
    scrollTop: $('#results-container').offset().top
  }, 600);

  // 2. Plot all the charts
  Plotly.newPlot('efficient-frontier-container', data.efficient_frontier_data, data.efficient_frontier_layout);
  Plotly.newPlot('heatmap-container', data.heatmap_data, data.heatmap_layout);
  Plotly.newPlot('nav-plot-monthly', data.portfolio_data.monthly, data.portfolio_layout.monthly);

  // 3. Build and render HTML tables from the JSON data
  // --- Key Metrics Table (mu/vol/sharpe) ---
  let muTable = '<table id="mu_table" class="results-table"><thead><tr><th>Ticker</th><th>Expected Return</th><th>Volatility</th><th>Sharpe Ratio</th></tr></thead><tbody>';
  for (const ticker in data.mu) {
      muTable += `<tr>
          <td>${ticker}</td>
          <td>${(data.mu[ticker] * 100).toFixed(2)}%</td>
          <td>${(data.vol[ticker] * 100).toFixed(2)}%</td>
          <td>${(data.sharpe[ticker]).toFixed(2)}</td>
      </tr>`;
  }
  muTable += '</tbody></table>';
  $('#mu-table-container').html(muTable);

  // --- Performance Summary Table ---
  let perfTable = '<table id="perf-summary" class="results-table"><thead><tr>';
  Object.keys(data.perf_dict[0]).forEach(key => perfTable += `<th>${key}</th>`);
  perfTable += '</tr></thead><tbody>';
  data.perf_dict.forEach(row => {
      perfTable += '<tr>';
      Object.values(row).forEach(val => perfTable += `<td>${(typeof val === 'number') ? val.toFixed(3) : val}</td>`);
      perfTable += '</tr>';
  });
  perfTable += '</tbody></table>';
  $('#perf-summary-container').html(perfTable);
  
  // --- **NEW**: Correlation Matrix Table ---
  const corrMatrix = data.corr_matrix;
  const tickers = Object.keys(corrMatrix);
  let corrTable = '<table id="correlation-matrix" class="results-table"><thead><tr><th>Ticker</th>';
  tickers.forEach(ticker => {
      corrTable += `<th>${ticker}</th>`;
  });
  corrTable += '</tr></thead><tbody>';
  tickers.forEach(ticker => {
      corrTable += `<tr><td><strong>${ticker}</strong></td>`;
      tickers.forEach(otherTicker => {
          corrTable += `<td>${corrMatrix[ticker][otherTicker].toFixed(3)}</td>`;
      });
      corrTable += '</tr>';
  });
  corrTable += '</tbody></table>';
  $('#correlation-matrix-container').html(corrTable);
}


// --- Bind Autocomplete and Handle Form Submission on Page Load ---
$(document).ready(function () {
  $('.ticker-input').each(function () {
    initializeAutocomplete(this);
  });

  $('form').on('submit', async function (event) {
    event.preventDefault();

    const submitButton = $(this).find('.submit-btn');
    submitButton.prop('disabled', true).text('Calculating...');
    $('#results-container').html('<div class="loader"></div>');
    $('#error-message').hide().text('');

    const formData = new FormData(this);
    const tickers = [];
    const minWeights = {};
    const maxWeights = {};
    for (let i = 0; i < formData.get('asset_count'); i++) {
        const ticker = formData.get(`tickers_${i}`);
        if (ticker) {
            tickers.push(ticker);
            minWeights[ticker] = formData.get(`min_${i}`);
            maxWeights[ticker] = formData.get(`max_${i}`);
        }
    }

    const requestBody = {
        tickers: tickers,
        start_date: formData.get('start_date'),
        end_date: formData.get('end_date'),
        target_return: formData.get('target_return'),
        target_volatility: formData.get('target_volatility'),
        min_weights: minWeights,
        max_weights: maxWeights
    };
    
    try {
        const response = await fetch('/api/portfolio_analysis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `API Error: ${response.statusText}`);
        }
        const data = await response.json();
        renderResults(data);
    } catch (error) {
        console.error("Error fetching or plotting data:", error);
        $('#results-container').html('');
        $('#error-message').text(`An error occurred: ${error.message}`).show();
    } finally {
        submitButton.prop('disabled', false).text('Run Optimization');
    }
  });
});