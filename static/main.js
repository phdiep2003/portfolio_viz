// --- NEW: A helper function that delays execution until after a pause ---
function debounce(func, delay = 250) {
  let timeoutId;
  return function(...args) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => {
      func.apply(this, args);
    }, delay);
  };
}

// --- NEW: A function to resize all Plotly charts ---
function resizePlotlyCharts() {
  const plotContainers = [
    'efficient-frontier-container',
    'heatmap-container',
    'nav-plot-monthly'
  ].map(id => document.getElementById(id)).filter(el => el && el.offsetParent !== null);

  plotContainers.forEach(container => {
    Plotly.Plots.resize(container);
  });
}


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
      // Manually trigger the input event to save the state
      $(this).trigger('input');
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

// --- MODIFIED: Add New Asset Row (now accepts data) ---
function addRow(asset = { ticker: '', min: '0', max: '15' }) {
  const tbody = document.getElementById('assets-tbody');
  const rowCount = tbody.rows.length;
  const newRow = document.createElement('tr');

  newRow.innerHTML = `
      <td>
          <input type="text" name="tickers_${rowCount}" class="ticker-input" autocomplete="off" data-index="${rowCount}" value="${asset.ticker}">
      </td>
      <td>
          <div class="input-with-symbol">
              <input type="number" name="min_${rowCount}" min="0" max="100" step="0.01" value="${asset.min}">
              <span class="percent-symbol">%</span>
          </div>
      </td>
      <td>
          <div class="input-with-symbol">
              <input type="number" name="max_${rowCount}" min="0" max="100" step="0.01" value="${asset.max}">
              <span class="percent-symbol">%</span>
          </div>
      </td>
  `;

  tbody.appendChild(newRow);
  document.getElementById('asset_count').value = rowCount + 1;
  initializeAutocomplete(newRow.querySelector('.ticker-input'));
}

// --- NEW: Save the entire form state to localStorage ---
function saveFormState() {
    const assets = [];
    $('#assets-tbody tr').each(function() {
        const row = $(this);
        assets.push({
            ticker: row.find('.ticker-input').val(),
            min: row.find('input[name^="min_"]').val(),
            max: row.find('input[name^="max_"]').val()
        });
    });

    const formData = {
        startDate: $('#start_date').val(),
        endDate: $('#end_date').val(),
        targetReturn: $('#target_return').val(),
        targetVolatility: $('#target_volatility').val(),
        assets: assets
    };

    localStorage.setItem('portfolioFormData', JSON.stringify(formData));
}

// --- NEW: Load form state from localStorage on page load ---
function loadFormState() {
    const savedData = localStorage.getItem('portfolioFormData');
    if (!savedData) {
        // If no data, initialize the default rows
        $('.ticker-input').each(function () {
            initializeAutocomplete(this);
        });
        return;
    }

    const data = JSON.parse(savedData);

    // Populate static fields
    $('#start_date').val(data.startDate);
    $('#end_date').val(data.endDate);
    $('#target_return').val(data.targetReturn);
    $('#target_volatility').val(data.targetVolatility);

    // Re-create dynamic asset rows
    const tbody = $('#assets-tbody');
    tbody.empty(); // Clear any hardcoded default rows

    if (data.assets && data.assets.length > 0) {
        data.assets.forEach(asset => addRow(asset));
    } else {
        // If no assets were saved, add one blank row to start
        addRow();
    }
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

function rateLimit(fn, interval = 5000) { // default: 1 request every 5s
  let lastTime = 0;
  return async function (...args) {
    const now = Date.now();
    if (now - lastTime < interval) {
      alert(`Please wait ${((interval - (now - lastTime)) / 1000).toFixed(1)}s before retrying.`);
      return;
    }
    lastTime = now;
    return fn.apply(this, args);
  };
}

// --- MODIFIED: Bind Events on Page Load ---
$(document).ready(function () {
  // 1. Load any previously saved state from localStorage
  loadFormState();
  
  // 2. Listen for window resize to make plots responsive
  window.addEventListener('resize', debounce(resizePlotlyCharts));

  // 3. Save form state whenever any input changes
  // We use event delegation to capture events on dynamically added rows
  $('form').on('input', 'input', saveFormState);

  // 4. Handle the main form submission
  $('form').on('submit', rateLimit(async function (event) {
    event.preventDefault();
    saveFormState();
    const submitButton = $(this).find('.submit-btn');

    // --- Loading effect ---
    submitButton.prop('disabled', true);
    let dotCount = 0;
    const loadingInterval = setInterval(() => {
        dotCount = (dotCount + 1) % 4;
        submitButton.text(`Calculating${'.'.repeat(dotCount)}`);
    }, 400);

    $('#results-container').html('<div class="loader"></div>');
    $('#error-message').hide().text('');

    const formData = new FormData(this);
    const tickers = [];
    const minWeights = {};
    const maxWeights = {};
    const assetCount = $('#assets-tbody tr').length;

    for (let i = 0; i < assetCount; i++) {
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
        clearInterval(loadingInterval);
        submitButton.prop('disabled', false).text('Run Optimization');
    }
  }, 5000)); // <-- 5 seconds between submissions
});
