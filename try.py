from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

sample_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'UNH']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        tickers = request.form.getlist('tickers[]')
        min_weights = request.form.getlist('min_weights[]')
        max_weights = request.form.getlist('max_weights[]')
        
        return f"""
        Received tickers: {tickers}<br>
        Min weights: {min_weights}<br>
        Max weights: {max_weights}
        """

    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Portfolio Input</title>
    <style>
        table, th, td { border: 1px solid black; border-collapse: collapse; padding: 5px; }
        th { background: #eee; }
        input[type=text], input[type=number] { width: 120px; }
        button { margin-top: 10px; }
    </style>
</head>
<body>

<h2>Portfolio Inputs</h2>
<form method="POST" action="/">
    <table id="assets-table">
        <thead>
            <tr>
                <th>Ticker</th>
                <th>Min Weight (%)</th>
                <th>Max Weight (%)</th>
                <th>Remove</th>
            </tr>
        </thead>
        <tbody id="assets-tbody">
            <tr>
                <td>
                    <input type="text" name="tickers[]" list="tickers-list" />
                    <datalist id="tickers-list">
                        {% for ticker in sample_tickers %}
                        <option value="{{ ticker }}">
                        {% endfor %}
                    </datalist>
                </td>
                <td>
                    <input type="number" name="min_weights[]" min="0" max="100" step="0.01" value="0" />
                </td>
                <td>
                    <input type="number" name="max_weights[]" min="0" max="100" step="0.01" value="15" />
                </td>
                <td><button type="button" class="remove-row">X</button></td>
            </tr>
        </tbody>
    </table>

    <button type="button" id="add-row">Add Asset</button><br><br>
    <button type="submit">Run Optimization</button>
</form>

<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script>
    function addRow() {
        const newRow = `
            <tr>
                <td>
                    <input type="text" name="tickers[]" list="tickers-list" />
                </td>
                <td>
                    <input type="number" name="min_weights[]" min="0" max="100" step="0.01" value="0" />
                </td>
                <td>
                    <input type="number" name="max_weights[]" min="0" max="100" step="0.01" value="15" />
                </td>
                <td><button type="button" class="remove-row">X</button></td>
            </tr>
        `;
        $('#assets-tbody').append(newRow);
    }

    $(document).ready(function() {
        // Add row button
        $('#add-row').click(function() {
            addRow();
        });

        // Remove row button
        $('#assets-tbody').on('click', '.remove-row', function() {
            $(this).closest('tr').remove();
        });
    });
</script>

</body>
</html>
    ''', sample_tickers=sample_tickers)

if __name__ == '__main__':
    app.run(debug=True)