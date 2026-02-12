document.addEventListener('DOMContentLoaded', () => {
    fetchStats();
    fetchFeatureImportance();
    fetchDataSample();
    fetchValidationReport();

    // Smooth Scrolling for Sidebar Links
    document.querySelectorAll('.sidebar nav a').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);

            if (targetElement) {
                // Update active state
                document.querySelectorAll('.sidebar nav li').forEach(li => li.classList.remove('active'));
                this.parentElement.classList.add('active');

                // Scroll the main content container
                targetElement.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
});

// Fetch Top Level Stats
async function fetchStats() {
    try {
        const response = await fetch('/api/spark/stats');
        const data = await response.json();

        document.getElementById('total-movies').textContent = data.total_movies || 0;
        document.getElementById('avg-revenue').textContent = formatCurrency(data.avg_revenue || 0);
        document.getElementById('avg-sentiment').textContent = (data.avg_sentiment || 0).toFixed(2);
    } catch (error) {
        console.error('Error fetching stats:', error);
    }
}

// Fetch Feature Importance and Render Chart
async function fetchFeatureImportance() {
    try {
        const response = await fetch('/api/model/importance');
        const data = await response.json();

        if (data.error) return;

        // Sort and take top 10
        const sorted = data.sort((a, b) => b.importance - a.importance).slice(0, 10);

        const xValues = sorted.map(i => i.feature);
        const yValues = sorted.map(i => i.importance);

        const trace = {
            x: yValues.reverse(), // Horizontal bar, so values on X
            y: xValues.reverse(),
            type: 'bar',
            orientation: 'h',
            marker: {
                color: '#00d2ff'
            }
        };

        const layout = {
            margin: { t: 20, l: 150 }
        };

        Plotly.newPlot('feature-importance-chart', [trace], layout);

    } catch (error) {
        console.error('Error fetching importance:', error);
    }
}

function renderSentimentChart(data) {
    const trace = {
        x: data,
        type: 'histogram',
        marker: {
            color: '#3a7bd5'
        },
        opacity: 0.7
    };

    const layout = {
        margin: { t: 20, l: 40, b: 40 },
        xaxis: { title: 'Sentiment Score' }
    };

    Plotly.newPlot('sentiment-chart', [trace], layout);
}

// Helper to format currency
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(value);
}

// Fetch Data Sample Table
async function fetchDataSample() {
    try {
        const response = await fetch('/api/data/sample');
        const data = await response.json();

        // Updated selector for simpler HTML
        const tbody = document.getElementById('data-sample-body');
        if (!tbody) return;

        tbody.innerHTML = '';

        // Sentiment Data for chart
        const sentiments = [];

        // LIMIT TO 20 MOVIES AS REQUESTED
        const limitedData = data.slice(0, 20);

        limitedData.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${row.title}</td>
                <td>${formatCurrency(row.budget)}</td>
                <td>${formatCurrency(row.revenue)}</td>
                <td>${(row.youtube_sentiment || 0).toFixed(2)}</td>
            `;
            tbody.appendChild(tr);

            // Only add to chart if sentiment is not 0
            if (row.youtube_sentiment !== 0) {
                sentiments.push(row.youtube_sentiment);
            }
        });

        // Render Sentiment Chart
        renderSentimentChart(sentiments);

    } catch (error) {
        console.error('Error fetching data sample:', error);
    }
}

// Fetch Validation Report
async function fetchValidationReport() {
    try {
        const response = await fetch('/api/validation/report');
        const data = await response.json();

        if (data.error) return;

        // Updated selector
        const tbody = document.getElementById('validation-table-body');
        if (!tbody) return;

        tbody.innerHTML = '';

        data.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${row.title}</td>
                <td>${formatCurrency(row.revenue)}</td>
                <td>${formatCurrency(row.predicted_revenue)}</td>
                <td>${row.error_pct.toFixed(2)}%</td>
            `;
            tbody.appendChild(tr);
        });

    } catch (error) {
        console.error('Error fetching validation report:', error);
    }
}
