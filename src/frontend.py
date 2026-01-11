"""
Frontend HTML template for the Housing Prices Predictor.
A clean, modern single-page interface for making predictions.
"""

FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Housing Price Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --secondary: #f1f5f9;
            --text: #1e293b;
            --text-light: #64748b;
            --success: #10b981;
            --warning: #f59e0b;
            --bg: #f8fafc;
            --card: #ffffff;
            --border: #e2e8f0;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem;
            color: var(--text);
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        
        .card {
            background: var(--card);
            border-radius: 16px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            color: white;
            padding: 2rem;
            text-align: center;
        }
        
        .header h1 {
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .header p {
            color: #94a3b8;
            font-size: 0.95rem;
        }
        
        .badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            background: rgba(99, 102, 241, 0.2);
            color: #a5b4fc;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 500;
            margin-top: 1rem;
        }
        
        .content {
            padding: 2rem;
        }
        
        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
        }
        
        @media (max-width: 600px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .form-group {
            display: flex;
            flex-direction: column;
        }
        
        .form-group label {
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text);
            margin-bottom: 0.5rem;
        }
        
        .form-group .hint {
            font-size: 0.75rem;
            color: var(--text-light);
            margin-bottom: 0.25rem;
        }
        
        .form-group input {
            padding: 0.75rem 1rem;
            border: 2px solid var(--border);
            border-radius: 8px;
            font-size: 1rem;
            transition: all 0.2s;
            font-family: inherit;
        }
        
        .form-group input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }
        
        .btn {
            display: block;
            width: 100%;
            padding: 1rem;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            margin-top: 1.5rem;
        }
        
        .btn:hover {
            background: var(--primary-dark);
            transform: translateY(-1px);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .btn:disabled {
            background: var(--text-light);
            cursor: not-allowed;
            transform: none;
        }
        
        .result {
            margin-top: 1.5rem;
            padding: 1.5rem;
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
            border-radius: 12px;
            text-align: center;
            display: none;
        }
        
        .result.show {
            display: block;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .result .label {
            font-size: 0.875rem;
            color: var(--text-light);
            margin-bottom: 0.5rem;
        }
        
        .result .price {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--success);
        }
        
        .result .meta {
            margin-top: 1rem;
            font-size: 0.75rem;
            color: var(--text-light);
        }
        
        .footer {
            text-align: center;
            padding: 1.5rem;
            border-top: 1px solid var(--border);
            color: var(--text-light);
            font-size: 0.875rem;
        }
        
        .footer a {
            color: var(--primary);
            text-decoration: none;
        }
        
        .footer a:hover {
            text-decoration: underline;
        }
        
        .tech-stack {
            display: flex;
            justify-content: center;
            gap: 0.5rem;
            margin-top: 0.75rem;
            flex-wrap: wrap;
        }
        
        .tech-badge {
            padding: 0.25rem 0.5rem;
            background: var(--secondary);
            border-radius: 4px;
            font-size: 0.7rem;
            color: var(--text-light);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <div class="header">
                <h1>Housing Price Predictor</h1>
                <p>Enter property details to get an instant price estimate</p>
                <span class="badge" id="mode-badge">Loading...</span>
            </div>
            
            <div class="content">
                <form id="predict-form">
                    <div class="form-grid">
                        <div class="form-group">
                            <label for="living_area">Living Area</label>
                            <span class="hint">Square feet above ground</span>
                            <input type="number" id="living_area" name="living_area" 
                                   value="1500" min="500" max="6000" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="quality">Overall Quality</label>
                            <span class="hint">Rating from 1-10</span>
                            <input type="number" id="quality" name="quality" 
                                   value="7" min="1" max="10" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="basement">Basement Area</label>
                            <span class="hint">Square feet</span>
                            <input type="number" id="basement" name="basement" 
                                   value="1000" min="0" max="3000">
                        </div>
                        
                        <div class="form-group">
                            <label for="garage">Garage Capacity</label>
                            <span class="hint">Number of cars</span>
                            <input type="number" id="garage" name="garage" 
                                   value="2" min="0" max="4">
                        </div>
                        
                        <div class="form-group">
                            <label for="year_built">Year Built</label>
                            <span class="hint">Construction year</span>
                            <input type="number" id="year_built" name="year_built" 
                                   value="2005" min="1900" max="2025">
                        </div>
                    </div>
                    
                    <button type="submit" class="btn" id="submit-btn">
                        Get Price Estimate
                    </button>
                </form>
                
                <div class="result" id="result">
                    <div class="label">Estimated Price</div>
                    <div class="price" id="price">$0</div>
                    <div class="meta" id="meta"></div>
                </div>
            </div>
            
            <div class="footer">
                <div>
                    <a href="/docs">API Documentation</a> | 
                    <a href="/health">Health Check</a> |
                    <a href="https://github.com" target="_blank">GitHub</a>
                </div>
                <div class="tech-stack">
                    <span class="tech-badge">ZenML</span>
                    <span class="tech-badge">MLflow</span>
                    <span class="tech-badge">FastAPI</span>
                    <span class="tech-badge">XGBoost</span>
                    <span class="tech-badge">GCP Cloud Run</span>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Check health on load
        fetch('/health')
            .then(res => res.json())
            .then(data => {
                const badge = document.getElementById('mode-badge');
                if (data.demo_mode) {
                    badge.textContent = 'Demo Mode';
                    badge.style.background = 'rgba(245, 158, 11, 0.2)';
                    badge.style.color = '#fbbf24';
                } else {
                    badge.textContent = 'Production Model';
                    badge.style.background = 'rgba(16, 185, 129, 0.2)';
                    badge.style.color = '#34d399';
                }
            })
            .catch(() => {
                document.getElementById('mode-badge').textContent = 'Offline';
            });
        
        // Form submission
        document.getElementById('predict-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const btn = document.getElementById('submit-btn');
            btn.disabled = true;
            btn.textContent = 'Calculating...';
            
            const features = {
                "Gr Liv Area": parseInt(document.getElementById('living_area').value),
                "Overall Qual": parseInt(document.getElementById('quality').value),
                "Total Bsmt SF": parseInt(document.getElementById('basement').value) || 0,
                "Garage Cars": parseInt(document.getElementById('garage').value) || 0,
                "Year Built": parseInt(document.getElementById('year_built').value)
            };
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ features })
                });
                
                const data = await response.json();
                
                const result = document.getElementById('result');
                const price = document.getElementById('price');
                const meta = document.getElementById('meta');
                
                price.textContent = '$' + data.predicted_price.toLocaleString();
                meta.textContent = `Model: ${data.model_version} | ${data.demo_mode ? 'Demo' : 'Production'}`;
                result.classList.add('show');
                
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Get Price Estimate';
            }
        });
    </script>
</body>
</html>
"""
