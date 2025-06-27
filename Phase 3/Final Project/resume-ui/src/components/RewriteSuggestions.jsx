import React, { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

const RewriteSuggestions = () => {
  const navigate = useNavigate();
  const location = useLocation();

  // 從第一頁傳來的 state 取得 payload
  const { payload } = location.state || {};

  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // const BASE = import.meta.env.VITE_API_BASE; 
  // 確保 .env.development 裡有 VITE_API_BASE=https://你的-ngrok-url 或 http://localhost:8000

  useEffect(() => {
    if (!payload) {
      setError('Missing payload');
      setLoading(false);
      return;
    }

    const fetchRewrite = async () => {
      try {
        const res = await fetch('/rewrite', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          const text = await res.text();
          throw new Error(`Rewrite API ${res.status}: ${text}`);
        }
        const data = await res.json();
        setItems(data.items || []);
      } catch (err) {
        console.error(err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchRewrite();
  }, [payload]);

  if (loading) {
    return <div className="text-center mt-5">Loading suggestions…</div>;
  }
  if (error) {
    return (
      <div className="text-center mt-5">
        <p className="text-danger">Error: {error}</p>
        <button className="btn btn-secondary" onClick={() => navigate('/')}>
          Back Home
        </button>
      </div>
    );
  }

  return (
    <div style={{ width: '100%', maxWidth: '600px', margin: '0 auto', padding: '2rem' }}>
      <h2 className="text-center mb-4">Step 3: Rewrite Suggestions</h2>

      {/* Work Experience */}
      <div className="mb-4">
        <h4>Work Experience</h4>
        {items.length === 0 && <p>No suggestions returned.</p>}
        {items.map((item, idx) => (
          <div key={idx} className="mb-3 p-3 border rounded">
            <p className="fw-bold mb-1">Original:</p>
            <p className="mb-2">{item.original}</p>
            <p className="fw-bold mb-1">Suggestion:</p>
            <p>{item.suggestion}</p>
          </div>
        ))}
      </div>

      <div className="text-center mt-4">
        <button
          className="btn btn-primary px-4"
          onClick={() => navigate('/')}
        >
          Done
        </button>
      </div>
    </div>
  );
};

export default RewriteSuggestions;