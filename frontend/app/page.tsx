'use client';

import { useState } from 'react';
import SearchBar from '../components/SearchBar';
import ResultsDisplay from '../components/ResultsDisplay';

export default function Home() {
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (query: string) => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      // Call Python backend directly (assumes Flask/FastAPI running on localhost:5000)
      const response = await fetch('http://localhost:8000/recommend', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query, top_k: 10 }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch recommendations');
      }

      const data = await response.json();
      console.log('API Response:', data);
      console.log('Recommendations:', data.recommendations);
      setResults(data.recommendations || []);
    } catch (err) {
      console.error('Error:', err);
      setError('Failed to get recommendations. Please check if the Python backend is running on port 8000.');
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-white">
      <header className="border-b border-gray-200 bg-white">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 items-center">
            <img src="/shl-logo.svg" alt="SHL" className="h-8" />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-12">
        <div className="mb-12 text-center">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">Assessment Recommendation System</h2>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            Find the perfect assessments for your hiring needs. Describe your requirements and get AI-powered recommendations.
          </p>
        </div>

        <div className="mb-12">
          <SearchBar onSearch={handleSearch} loading={loading} />
        </div>

        {error && (
          <div className="mb-8 rounded-lg bg-red-50 p-4 text-red-800">
            <p className="text-sm font-medium">{error}</p>
          </div>
        )}

        <ResultsDisplay results={results} loading={loading} />
      </main>

      <footer className="mt-20 border-t border-gray-200 bg-gray-50">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
          <p className="text-center text-sm text-gray-500">
            © 2025 SHL Assessment Recommendation System. Powered by Multi-Vector Retrieval V2.
          </p>
        </div>
      </footer>
    </div>
  );
}
