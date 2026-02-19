document.addEventListener('DOMContentLoaded', () => {
    fetchStats();
    fetchModelScores();
    fetchFeatureImportance();
    fetchValidationReport();

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
                sentimentHelper.style.color = "#10b981";
            } else if (val <= 3.0) {
                sentimentHelper.textContent = "Negative Influence: Penalizes Revenue (-20%)";
                sentimentHelper.style.color = "#ef4444";
            } else {
                sentimentHelper.textContent = "Neutral Impact";
                sentimentHelper.style.color = "#94a3b8";
            }
        });
    }

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

                // Cast & Crew Inputs
                avg_cast_popularity: parseFloat(document.getElementById('avg_cast_popularity').value),
                avg_director_popularity: parseFloat(document.getElementById('avg_director_popularity').value),
                num_cast_members: parseInt(document.getElementById('num_cast_members').value),
                num_composers: parseInt(document.getElementById('num_composers').value),

                // Derived/Defaults
                max_cast_popularity: parseFloat(document.getElementById('avg_cast_popularity').value) * 1.5, // Estimate
                star_power_score: parseFloat(document.getElementById('avg_cast_popularity').value) * parseInt(document.getElementById('num_cast_members').value),
                max_director_popularity: parseFloat(document.getElementById('avg_director_popularity').value),
                director_experience_score: parseFloat(document.getElementById('avg_director_popularity').value),

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

        document.getElementById('total-movies').textContent = data.total_movies.toLocaleString();
        document.getElementById('avg-revenue').textContent = '$' + (data.avg_revenue / 1000000).toFixed(1) + 'M';
        document.getElementById('avg-sentiment').textContent = data.avg_sentiment.toFixed(2);
    } catch (e) {
        console.error('Error fetching stats:', e);
    }
}

async function fetchFeatureImportance() {
    try {
        const response = await fetch('/api/model/importance');
        const data = await response.json();

        if (data.error) return;

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

        Plotly.newPlot('feature-importance-chart', [trace], layout);
    } catch (e) {
        console.error(e);
    }
}

async function fetchValidationReport() {
    try {
        const response = await fetch('/api/validation/report');
        const data = await response.json();

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
    } catch (e) {
        console.error('Error validation report:', e);
    }
}

async function fetchModelScores() {
    try {
        const response = await fetch('/api/model/scores');
        const data = await response.json();
        if (data.error) { console.error(data.error); return; }

        document.getElementById('r2-score').textContent = data.r2_score.toFixed(4);
        document.getElementById('mae-score').textContent = '$' + (data.mae / 1e6).toFixed(1) + 'M';
        document.getElementById('mape-score').textContent = data.overall_mape.toFixed(1) + '%';
        document.getElementById('top50-mape').textContent = data.top50_mape.toFixed(1) + '%';
        document.getElementById('model-type').textContent = data.model_type;
        document.getElementById('total-features').textContent = data.total_features;
        document.getElementById('rmse-score').textContent = '$' + (data.rmse / 1e6).toFixed(1) + 'M';

        // Feature groups
        const groupsDiv = document.getElementById('feature-groups');
        if (groupsDiv && data.feature_groups) {
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
    }
}
