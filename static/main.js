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
