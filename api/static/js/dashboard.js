document.addEventListener('DOMContentLoaded', () => {
    fetchStats();
    fetchModelScores();
    fetchFeatureImportance();
    fetchValidationReport();
    fetchChartsData();
    fetchMoviePreview();

    // Sentiment Slider
    const sentimentSlider = document.getElementById('youtube_sentiment');
    const sentimentVal = document.getElementById('sentiment-val');
    const sentimentHelper = document.getElementById('sentiment-helper');

    if (sentimentSlider) {
        sentimentSlider.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            sentimentVal.textContent = val;
            if (val >= 9.0) {
                sentimentHelper.textContent = "High Influence: Boosts Revenue (+15%)";
                sentimentHelper.style.color = "#059669";
            } else if (val <= 3.0) {
                sentimentHelper.textContent = "Negative Influence: Penalizes Revenue (-20%)";
                sentimentHelper.style.color = "#dc2626";
            } else {
                sentimentHelper.textContent = "Neutral Impact";
                sentimentHelper.style.color = "#64748b";
            }
        });
    }

    // Sidebar active link on scroll
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.sidebar a');
    window.addEventListener('scroll', () => {
        let current = '';
        sections.forEach(s => {
            if (window.scrollY >= s.offsetTop - 120) current = s.id;
        });
        navLinks.forEach(a => {
            a.classList.toggle('active', a.getAttribute('href') === '#' + current);
        });
    });

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
                budget: parseFloat(document.getElementById('budget').value),
                overview: document.getElementById('overview').value,
                primary_genre: document.getElementById('primary_genre').value,
                release_month: parseInt(document.getElementById('release_month').value),
                youtube_sentiment: parseFloat(document.getElementById('youtube_sentiment').value),
                // Cast & crew: sent as 0 — backend fills them from the overview via CastLookup
                avg_cast_popularity: 0,
                avg_director_popularity: 0,
                num_cast_members: 0,
                num_composers: 0,
                max_cast_popularity: 0,
                star_power_score: 0,
                max_director_popularity: 0,
                director_experience_score: 0,
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
                if (!response.ok) throw new Error('Prediction failed');
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

function displayPrediction(result) {
    const resultCard = document.getElementById('prediction-result');
    resultCard.style.display = 'block';
    const revenueFormatted = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(result.predicted_revenue);
    document.getElementById('pred-revenue').textContent = revenueFormatted;
    document.getElementById('pred-class').textContent = result.success_class;
    document.getElementById('pred-conf').textContent = (result.confidence_score * 100).toFixed(1) + '%';

    const classElem = document.getElementById('pred-class');
    if (result.success_class === 'Hit' || result.success_class === 'Blockbuster') classElem.style.color = '#059669';
    else if (result.success_class === 'Flop') classElem.style.color = '#dc2626';
    else classElem.style.color = '#f59e0b';

    const nlpSection = document.getElementById('nlp-results');
    if (result.debug_info && result.debug_info.nlp_analysis) {
        nlpSection.style.display = 'block';
        const nlp = result.debug_info.nlp_analysis;
        const sentimentElem = document.getElementById('nlp-sentiment');
        sentimentElem.textContent = `${nlp.sentiment_label} (${nlp.sentiment_score.toFixed(2)})`;
        if (nlp.sentiment_label === 'Positive') sentimentElem.style.color = '#059669';
        else if (nlp.sentiment_label === 'Negative') sentimentElem.style.color = '#dc2626';
        else sentimentElem.style.color = '#64748b';
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
        document.getElementById('total-movies').textContent = (data.total_movies || 0).toLocaleString();
        document.getElementById('avg-revenue').textContent = '$' + ((data.avg_revenue || 0) / 1000000).toFixed(1) + 'M';
        document.getElementById('avg-sentiment').textContent = (data.avg_sentiment || 0).toFixed(2);
    } catch (e) { console.error('Error fetching stats:', e); }
}

async function fetchFeatureImportance() {
    try {
        const response = await fetch('/api/model/importance');
        const data = await response.json();
        if (data.error || !Array.isArray(data)) return;

        const top = data.slice(0, 15);
        const x = top.map(d => d.feature);
        const y = top.map(d => d.importance);

        Plotly.newPlot('feature-importance-chart', [{
            x, y,
            type: 'bar',
            marker: {
                color: y.map((_, i) => `hsl(${220 + i * 6}, 70%, ${50 + i * 1.5}%)`),
                line: { width: 0 }
            }
        }], {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            xaxis: { tickangle: -38, color: '#64748b', gridcolor: '#e2e8f0', tickfont: { size: 11 } },
            yaxis: { color: '#64748b', gridcolor: '#e2e8f0', gridwidth: 1 },
            margin: { t: 10, b: 110, l: 50, r: 10 },
            height: 320
        }, { responsive: true, displayModeBar: false });
    } catch (e) { console.error(e); }
}

async function fetchValidationReport() {
    try {
        const response = await fetch('/api/validation/report');
        const data = await response.json();

        let results = Array.isArray(data) ? data : (data.results || []);
        let total = Array.isArray(data) ? data.length : (data.total_count || results.length);

        const totalElem = document.getElementById('validation-total');
        if (totalElem) totalElem.textContent = total;

        const tbody = document.getElementById('validation-table-body');
        tbody.innerHTML = '';
        results.forEach(row => {
            const actual = row.actual_revenue || 0;
            const predicted = row.predicted_revenue || 0;
            const error = row.error_percentage || 0;
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${row.title}</td>
                <td>$${(actual / 1e6).toFixed(1)}M</td>
                <td>$${(predicted / 1e6).toFixed(1)}M</td>
                <td style="color:${error < 20 ? '#059669' : '#dc2626'};font-weight:500">${error.toFixed(1)}%</td>
            `;
            tbody.appendChild(tr);
        });
        renderAccuracyCharts(results);
    } catch (e) { console.error('Validation error:', e); }
}

async function fetchModelScores() {
    try {
        const response = await fetch('/api/model/scores');
        const data = await response.json();
        if (data.error) return;

        document.getElementById('r2-score').textContent = data.r2_score.toFixed(4);
        document.getElementById('mae-score').textContent = '$' + (data.mae / 1e6).toFixed(1) + 'M';
        document.getElementById('mape-score').textContent = data.overall_mape.toFixed(1) + '%';
        document.getElementById('top50-mape').textContent = data.top50_mape.toFixed(1) + '%';
        document.getElementById('model-type').textContent = data.model_type;
        document.getElementById('total-features').textContent = data.total_features;
        document.getElementById('rmse-score').textContent = '$' + (data.rmse / 1e6).toFixed(1) + 'M';

        const groupsDiv = document.getElementById('feature-groups');
        if (groupsDiv && data.feature_groups) {
            const colors = { financial: '#f59e0b', audience: '#3b82f6', trailer: '#ef4444', sentiment: '#8b5cf6', cast_crew: '#059669', engineered: '#06b6d4' };
            for (const [group, features] of Object.entries(data.feature_groups)) {
                const c = colors[group] || '#64748b';
                const tag = document.createElement('div');
                tag.style.cssText = `background:${c}15;border:1px solid ${c}33;border-radius:8px;padding:8px 14px;font-size:0.82rem;`;
                tag.innerHTML = `<strong style="color:${c}">${group.replace('_', ' ').toUpperCase()}</strong>: ${features.join(', ')}`;
                groupsDiv.appendChild(tag);
            }
        }
    } catch (e) { console.error('Model scores error:', e); }
}

// ── PLOTLY SHARED CONFIG ──────────────────────────────────────────────────

const PLOTLY_LIGHT = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Inter, sans-serif', size: 12, color: '#64748b' },
    margin: { t: 10, b: 60, l: 55, r: 15 },
    height: 280
};

// ── PREDICTION ACCURACY CHARTS ────────────────────────────────────────────

function renderAccuracyCharts(results) {
    if (!results || results.length === 0) return;

    const actuals = results.map(r => r.actual_revenue / 1e6);
    const predicted = results.map(r => r.predicted_revenue / 1e6);
    const errors = results.map(r => r.error_percentage);
    const titles = results.map(r => r.title);

    // Diagonal reference line (perfect prediction)
    const maxVal = Math.max(...actuals, ...predicted);
    const refLine = {
        x: [0, maxVal], y: [0, maxVal],
        mode: 'lines', type: 'scatter', name: 'Perfect Prediction',
        line: { color: '#94a3b8', width: 1.5, dash: 'dash' },
        hoverinfo: 'skip'
    };

    // Color-code points by error bracket
    const pointColors = errors.map(e =>
        e < 15 ? '#22c55e' : e < 40 ? '#f59e0b' : '#ef4444');

    const scatter = {
        x: actuals, y: predicted,
        mode: 'markers', type: 'scatter',
        name: 'Movies',
        text: titles.map((t, i) =>
            `${t}<br>Actual: $${actuals[i].toFixed(1)}M<br>Predicted: $${predicted[i].toFixed(1)}M<br>Error: ${errors[i].toFixed(1)}%`),
        hovertemplate: '%{text}<extra></extra>',
        marker: {
            color: pointColors, size: 8, opacity: 0.8,
            line: { color: '#fff', width: 0.8 }
        }
    };

    Plotly.newPlot('chart-actual-vs-predicted', [refLine, scatter], {
        ...PLOTLY_LIGHT,
        height: 300,
        xaxis: { title: 'Actual Revenue ($M)', gridcolor: '#f1f5f9', color: '#64748b', zeroline: false },
        yaxis: { title: 'Predicted Revenue ($M)', gridcolor: '#f1f5f9', color: '#64748b', zeroline: false },
        legend: { orientation: 'h', y: -0.25, font: { size: 11 } },
        annotations: [{
            x: 0.97, y: 0.97, xref: 'paper', yref: 'paper',
            text: `<b>${errors.filter(e => e < 20).length}/${results.length}</b> within 20% error`,
            showarrow: false, font: { size: 11, color: '#22c55e' },
            align: 'right'
        }]
    }, { responsive: true, displayModeBar: false });

    // Error distribution histogram
    const buckets = [0, 10, 20, 30, 40, 50, 75, 100, 200];
    const labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-75%', '75-100%', '100%+'];
    const counts = buckets.slice(0, -1).map((lo, i) =>
        errors.filter(e => e >= lo && e < buckets[i + 1]).length);
    const barColors = ['#22c55e', '#4ade80', '#86efac', '#f59e0b', '#fb923c', '#ef4444', '#b91c1c', '#7f1d1d'];

    Plotly.newPlot('chart-error-dist', [{
        x: labels, y: counts,
        type: 'bar',
        marker: { color: barColors, line: { width: 0 } },
        hovertemplate: '<b>%{x}</b><br>%{y} movies<extra></extra>'
    }], {
        ...PLOTLY_LIGHT,
        height: 300,
        xaxis: { color: '#64748b', gridcolor: '#f1f5f9', tickangle: -20 },
        yaxis: { title: 'Movies', gridcolor: '#f1f5f9', color: '#64748b' }
    }, { responsive: true, displayModeBar: false });
}

async function fetchChartsData() {
    try {
        const res = await fetch('/api/charts/data');
        const d = await res.json();
        if (d.error) return;

        // 1. Revenue by Genre
        if (d.revenue_by_genre) {
            const genreColors = d.revenue_by_genre.genres.map((_, i) =>
                `hsl(${210 + i * 12}, 65%, 55%)`);
            Plotly.newPlot('chart-genre-revenue', [{
                x: d.revenue_by_genre.genres,
                y: d.revenue_by_genre.avg_revenues,
                type: 'bar',
                marker: { color: genreColors, line: { width: 0 } },
                hovertemplate: '<b>%{x}</b><br>Avg Revenue: $%{y}M<extra></extra>'
            }], {
                ...PLOTLY_LIGHT,
                xaxis: { tickangle: -30, gridcolor: '#f1f5f9', color: '#64748b' },
                yaxis: { title: 'Avg Revenue ($M)', gridcolor: '#f1f5f9', color: '#64748b' }
            }, { responsive: true, displayModeBar: false });
        }

        // 2. Budget vs Revenue scatter
        if (d.budget_vs_revenue) {
            const genres = [...new Set(d.budget_vs_revenue.map(m => m.primary_genre))];
            const palette = ['#3b82f6', '#ef4444', '#f59e0b', '#22c55e', '#a855f7', '#ec4899', '#06b6d4', '#f97316', '#8b5cf6', '#64748b'];
            const traces = genres.map((g, i) => {
                const pts = d.budget_vs_revenue.filter(m => m.primary_genre === g);
                return {
                    x: pts.map(m => m.budget / 1e6),
                    y: pts.map(m => m.revenue / 1e6),
                    mode: 'markers',
                    type: 'scatter',
                    name: g,
                    text: pts.map(m => m.title),
                    marker: { size: 7, color: palette[i % palette.length], opacity: 0.75 },
                    hovertemplate: '<b>%{text}</b><br>Budget: $%{x}M<br>Revenue: $%{y}M<extra></extra>'
                };
            });
            Plotly.newPlot('chart-budget-revenue', traces, {
                ...PLOTLY_LIGHT,
                xaxis: { title: 'Budget ($M)', gridcolor: '#f1f5f9', color: '#64748b' },
                yaxis: { title: 'Revenue ($M)', gridcolor: '#f1f5f9', color: '#64748b' },
                legend: { orientation: 'h', y: -0.3, font: { size: 10 } },
                margin: { ...PLOTLY_LIGHT.margin, b: 90 }
            }, { responsive: true, displayModeBar: false });
        }

        // 3. Monthly distribution
        if (d.monthly_distribution) {
            Plotly.newPlot('chart-monthly', [{
                x: d.monthly_distribution.months,
                y: d.monthly_distribution.counts,
                type: 'bar',
                marker: {
                    color: d.monthly_distribution.counts,
                    colorscale: [[0, '#bfdbfe'], [0.5, '#3b82f6'], [1, '#1e3a8a']],
                    showscale: false,
                    line: { width: 0 }
                },
                hovertemplate: '<b>%{x}</b><br>%{y} movies<extra></extra>'
            }], {
                ...PLOTLY_LIGHT,
                xaxis: { color: '#64748b', gridcolor: '#f1f5f9' },
                yaxis: { title: 'Movies', gridcolor: '#f1f5f9', color: '#64748b' }
            }, { responsive: true, displayModeBar: false });
        }

        // 4. Success class pie
        if (d.success_distribution) {
            const pieColors = ['#22c55e', '#3b82f6', '#f59e0b', '#ef4444', '#94a3b8'];
            Plotly.newPlot('chart-success', [{
                labels: d.success_distribution.labels,
                values: d.success_distribution.values,
                type: 'pie',
                marker: { colors: pieColors, line: { color: '#fff', width: 2 } },
                textinfo: 'label+percent',
                textfont: { size: 12 },
                hovertemplate: '<b>%{label}</b><br>%{value} movies (%{percent})<extra></extra>'
            }], {
                ...PLOTLY_LIGHT,
                margin: { t: 10, b: 10, l: 10, r: 10 },
                legend: { orientation: 'h', y: -0.15, font: { size: 11 } }
            }, { responsive: true, displayModeBar: false });
        }

    } catch (e) { console.error('Charts error:', e); }
}

// ── NEW: Movie Preview Section ─────────────────────────────────────────────

function getGenreColor(genre) {
    const colors = {
        'Action': '#ef4444', 'Adventure': '#f97316', 'Comedy': '#eab308',
        'Drama': '#22c55e', 'Horror': '#a855f7', 'Science Fiction': '#3b82f6',
        'Thriller': '#ec4899', 'Animation': '#06b6d4', 'Romance': '#f43f5e',
        'Crime': '#64748b', 'Fantasy': '#8b5cf6', 'Mystery': '#6366f1'
    };
    return colors[genre] || '#64748b';
}

async function fetchMoviePreview() {
    try {
        const res = await fetch('/api/movies?limit=12&sort=revenue');
        const data = await res.json();
        if (data.error || !data.movies) return;

        const grid = document.getElementById('movie-preview-grid');
        if (!grid) return;
        grid.innerHTML = '';

        data.movies.forEach(m => {
            const color = getGenreColor(m.primary_genre);
            const revenue = m.revenue >= 1e6 ? '$' + (m.revenue / 1e6).toFixed(1) + 'M' : (m.revenue > 0 ? '$' + Math.round(m.revenue).toLocaleString() : '—');
            const card = document.createElement('div');
            card.className = 'preview-card';
            card.innerHTML = `
                <div class="preview-card-title">${m.title || 'Unknown'}</div>
                <span class="genre-badge-sm" style="background:${color}18;color:${color};border:1px solid ${color}33">${m.primary_genre || 'N/A'}</span>
                <div class="preview-revenue">${revenue}</div>
                <div class="preview-sentiment">Sentiment: ${m.youtube_sentiment ? m.youtube_sentiment.toFixed(2) : '—'}</div>
            `;
            grid.appendChild(card);
        });
    } catch (e) { console.error('Movie preview error:', e); }
}
