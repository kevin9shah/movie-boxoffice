document.addEventListener('DOMContentLoaded', () => {
    fetchStats();
    fetchModelScores();
    fetchFeatureImportance();
    fetchValidationReport();

    // Prediction Form
    const form = document.getElementById('prediction-form');
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            const btn = form.querySelector('.btn-predict');
            const originalText = btn.textContent;
            btn.textContent = 'Predicting...';
            btn.disabled = true;

            const movieData = {
                title: document.getElementById('title').value,
                budget: numOrDefault('budget', 0),
                runtime: Math.trunc(numOrDefault('runtime', 120)),
                overview: document.getElementById('overview').value,
                primary_genre: document.getElementById('primary_genre').value || 'Action',

                // Defaults for demo
                popularity: 50.0,
                vote_average: 7.0,
                vote_count: 1000,
                trailer_views: 1000000,
                trailer_likes: 50000,
                trailer_comments: 1000,
                trailer_popularity_index: 0.8,
                interaction_rate: 0.05,
                engagement_velocity: 0.1,
                sentiment_volatility: 0.5,
                trend_momentum: 50.0
            };

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(movieData)
                });

                if (!response.ok) {
                    let detail = 'Prediction failed';
                    try {
                        const errData = await response.json();
                        detail = errData.detail || errData.error || detail;
                    } catch (_) {}
                    throw new Error(detail);
                }

                const result = await response.json();
                displayPrediction(result);
            } catch (error) {
                console.error(error);
                alert('Error making prediction: ' + error.message);
            } finally {
                btn.textContent = originalText;
                btn.disabled = false;
            }
        });
    }
});

function safeSetText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function numOrDefault(id, fallback = 0) {
    const el = document.getElementById(id);
    if (!el) return fallback;
    const raw = el.value;
    if (raw === '' || raw === null || raw === undefined) return fallback;
    const n = Number(raw);
    return Number.isFinite(n) ? n : fallback;
}

function displayPrediction(result) {
    const resultCard = document.getElementById('prediction-result');
    resultCard.style.display = 'block';

    // Format Revenue
    const revenueFormatted = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(result.predicted_revenue);

    document.getElementById('pred-revenue').textContent = revenueFormatted;
    document.getElementById('pred-class').textContent = result.success_class;
    document.getElementById('pred-conf').textContent = (result.confidence_score * 100).toFixed(1) + '%';

    // Style Class
    const classElem = document.getElementById('pred-class');
    if (result.success_class === 'Hit') classElem.style.color = '#10b981';
    else if (result.success_class === 'Flop') classElem.style.color = '#ef4444';
    else classElem.style.color = '#fbbf24';

    // NLP Analysis Display
    const nlpSection = document.getElementById('nlp-results');
    if (result.debug_info && result.debug_info.nlp_analysis) {
        nlpSection.style.display = 'block';
        const nlp = result.debug_info.nlp_analysis;

        const sentimentElem = document.getElementById('nlp-sentiment');
        sentimentElem.textContent = `${nlp.sentiment_label} (${nlp.sentiment_score.toFixed(2)})`;
        if (nlp.sentiment_label === 'Positive') sentimentElem.style.color = '#10b981';
        else if (nlp.sentiment_label === 'Negative') sentimentElem.style.color = '#ef4444';
        else sentimentElem.style.color = '#94a3b8';

        document.getElementById('nlp-keywords').textContent = nlp.keywords.join(', ');
    } else {
        nlpSection.style.display = 'none';
    }

    resultCard.scrollIntoView({ behavior: 'smooth' });
}

async function fetchStats() {
    try {
        const response = await fetch('/api/spark/stats');
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        safeSetText('total-movies', Number(data.total_movies || 0).toLocaleString());
        safeSetText('avg-revenue', '$' + ((Number(data.avg_revenue || 0) / 1000000).toFixed(1)) + 'M');
        safeSetText('avg-sentiment', Number(data.avg_sentiment || 0).toFixed(2));
    } catch (e) {
        console.error('Error fetching stats:', e);
        safeSetText('total-movies', 'N/A');
        safeSetText('avg-revenue', 'N/A');
        safeSetText('avg-sentiment', 'N/A');
    }
}

async function fetchFeatureImportance() {
    try {
        const response = await fetch('/api/model/importance');
        const data = await response.json();

        if (data.error || !Array.isArray(data) || data.length === 0) {
            safeSetText('feature-importance-chart', 'Feature importance unavailable');
            return;
        }

        const x = data.map(d => d.feature);
        const y = data.map(d => d.importance);

        const trace = {
            x: x.slice(0, 15), // Top 15
            y: y.slice(0, 15),
            type: 'bar',
            marker: { color: '#3b82f6' }
        };

        const layout = {
            title: '',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            xaxis: { tickangle: -45, color: '#94a3b8' },
            yaxis: { color: '#94a3b8' },
            margin: { t: 10, b: 100 }
        };

        if (window.Plotly) {
            Plotly.newPlot('feature-importance-chart', [trace], layout);
        } else {
            safeSetText('feature-importance-chart', 'Plot library failed to load');
        }
    } catch (e) {
        console.error(e);
        safeSetText('feature-importance-chart', 'Feature importance unavailable');
    }
}

async function fetchValidationReport() {
    try {
        const response = await fetch('/api/validation/report');
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        // Handle wrapped response (new) or direct list (old compatibility)
        let results = [];
        let total = 0;

        if (Array.isArray(data)) {
            results = data;
            total = data.length;
        } else {
            results = data.results || [];
            total = data.total_count || results.length;
        }

        const totalElem = document.getElementById('validation-total');
        if (totalElem) totalElem.textContent = total;

        const tbody = document.getElementById('validation-table-body');
        tbody.innerHTML = '';

        results.forEach(row => {
            const tr = document.createElement('tr');
            // Check for valid properties before rendering
            const actual = row.actual_revenue || 0;
            const predicted = row.predicted_revenue || 0;
            const error = row.error_percentage || 0;

            tr.innerHTML = `
                <td>${row.title}</td>
                <td>$${(actual / 1000000).toFixed(1)}M</td>
                <td>$${(predicted / 1000000).toFixed(1)}M</td>
                <td style="color: ${error < 20 ? '#10b981' : '#ef4444'}">${error.toFixed(1)}%</td>
            `;
            tbody.appendChild(tr);
        });

        renderErrorCharts(results);
    } catch (e) {
        console.error('Error validation report:', e);
        safeSetText('error-distribution-chart', 'Error analytics unavailable');
        safeSetText('error-vs-revenue-chart', 'Error analytics unavailable');
        safeSetText('top-error-movies-chart', 'Error analytics unavailable');
    }
}

function renderErrorCharts(results) {
    if (!window.Plotly) {
        safeSetText('error-distribution-chart', 'Plot library failed to load');
        safeSetText('error-vs-revenue-chart', 'Plot library failed to load');
        safeSetText('top-error-movies-chart', 'Plot library failed to load');
        return;
    }

    if (!Array.isArray(results) || results.length === 0) {
        safeSetText('error-distribution-chart', 'No validation data');
        safeSetText('error-vs-revenue-chart', 'No validation data');
        safeSetText('top-error-movies-chart', 'No validation data');
        return;
    }

    const clean = results
        .map(r => ({
            title: r.title || 'Unknown',
            actual: Number(r.actual_revenue || 0),
            predicted: Number(r.predicted_revenue || 0),
            error: Number(r.error_percentage || 0)
        }))
        .filter(r => Number.isFinite(r.actual) && Number.isFinite(r.error));

    const errors = clean.map(r => r.error);
    Plotly.newPlot('error-distribution-chart', [{
        x: errors,
        type: 'histogram',
        nbinsx: 30,
        marker: { color: '#ef4444', opacity: 0.85 }
    }], {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 10, r: 10, b: 50, l: 50 },
        xaxis: { title: 'Error %', color: '#94a3b8' },
        yaxis: { title: 'Movie Count', color: '#94a3b8' }
    });

    const sample = clean.slice(0, 400);
    Plotly.newPlot('error-vs-revenue-chart', [{
        x: sample.map(r => r.actual / 1e6),
        y: sample.map(r => r.error),
        mode: 'markers',
        type: 'scatter',
        text: sample.map(r => r.title),
        marker: {
            color: sample.map(r => r.error),
            colorscale: 'YlOrRd',
            size: 8,
            opacity: 0.8,
            colorbar: { title: 'Err %' }
        }
    }], {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 10, r: 10, b: 50, l: 50 },
        xaxis: { title: 'Actual Revenue (USD Millions)', color: '#94a3b8' },
        yaxis: { title: 'Error %', color: '#94a3b8' }
    });

    const topErr = [...clean]
        .sort((a, b) => b.error - a.error)
        .slice(0, 15)
        .reverse();

    Plotly.newPlot('top-error-movies-chart', [{
        x: topErr.map(r => r.error),
        y: topErr.map(r => r.title),
        orientation: 'h',
        type: 'bar',
        marker: { color: '#f97316' }
    }], {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 10, r: 10, b: 30, l: 200 },
        xaxis: { title: 'Error %', color: '#94a3b8' },
        yaxis: { color: '#94a3b8' }
    });
}

async function fetchModelScores() {
    try {
        const response = await fetch('/api/model/scores');
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        const r2Text = (data.r2_score === null || data.r2_score === undefined) ? 'N/A' : Number(data.r2_score).toFixed(4);
        safeSetText('r2-score', r2Text);
        safeSetText('mae-score', '$' + (Number(data.mae || 0) / 1e6).toFixed(1) + 'M');
        safeSetText('mape-score', Number(data.overall_mape || 0).toFixed(1) + '%');
        safeSetText('top50-mape', Number(data.top50_mape || 0).toFixed(1) + '%');
        safeSetText('model-type', data.model_type || 'Unknown');
        safeSetText('total-features', String(data.total_features || 0));
        safeSetText('rmse-score', '$' + (Number(data.rmse || 0) / 1e6).toFixed(1) + 'M');

        // Feature groups
        const groupsDiv = document.getElementById('feature-groups');
        if (groupsDiv && data.feature_groups) {
            groupsDiv.innerHTML = '';
            const colors = { financial: '#f59e0b', audience: '#3b82f6', trailer: '#ef4444', sentiment: '#8b5cf6', cast_crew: '#22c55e', engineered: '#06b6d4' };
            for (const [group, features] of Object.entries(data.feature_groups)) {
                const tag = document.createElement('div');
                tag.style.cssText = `background: ${colors[group] || '#64748b'}22; border: 1px solid ${colors[group] || '#64748b'}; border-radius: 8px; padding: 8px 14px; font-size: 0.82rem;`;
                tag.innerHTML = `<strong style="color:${colors[group]}">${group.replace('_', ' ').toUpperCase()}</strong>: ${features.join(', ')}`;
                groupsDiv.appendChild(tag);
            }
        }
    } catch (e) {
        console.error('Error fetching model scores:', e);
        safeSetText('r2-score', 'N/A');
        safeSetText('mae-score', 'N/A');
        safeSetText('mape-score', 'N/A');
        safeSetText('top50-mape', 'N/A');
        safeSetText('model-type', 'Unavailable');
        safeSetText('total-features', 'N/A');
        safeSetText('rmse-score', 'N/A');
    }
}
