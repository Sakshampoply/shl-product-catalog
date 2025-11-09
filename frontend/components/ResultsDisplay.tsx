'use client';

interface Assessment {
  name: string;
  url: string;
  description?: string;
  test_types?: string[];
  duration_minutes?: number;  // Changed from duration: string
  job_levels?: string[];
  score?: number;
}

interface ResultsDisplayProps {
  results: Assessment[];
  loading: boolean;
}

export default function ResultsDisplay({ results, loading }: ResultsDisplayProps) {
  if (loading) {
    return (
      <div className="flex justify-center items-center py-12">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-600"></div>
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div className="text-center py-12">
        <svg
          className="mx-auto h-12 w-12 text-gray-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
        <h3 className="mt-2 text-sm font-medium text-gray-900">No results yet</h3>
        <p className="mt-1 text-sm text-gray-500">
          Enter a query above to find assessment recommendations
        </p>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-6">
        <h3 className="text-xl font-semibold text-gray-900">
          Top {results.length} Recommended Assessments
        </h3>
        <p className="mt-1 text-sm text-gray-500">
          Results ranked by relevance to your query
        </p>
      </div>

      <div className="space-y-4">
        {results.map((assessment, index) => (
          <div
            key={index}
            className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm hover:shadow-md transition-shadow"
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-2">
                  <span className="flex h-8 w-8 items-center justify-center rounded-full bg-green-100 text-sm font-semibold text-green-700">
                    {index + 1}
                  </span>
                  <h4 className="text-lg font-semibold text-gray-900">
                    {assessment.name}
                  </h4>
                </div>

                {assessment.description && (
                  <p className="mt-2 text-sm text-gray-600 line-clamp-2">
                    {assessment.description}
                  </p>
                )}

                <div className="mt-4 flex flex-wrap gap-2">
                  {assessment.test_types && assessment.test_types.length > 0 && (
                    <div className="flex items-center gap-2">
                      {assessment.test_types.map((type, i) => (
                        <span
                          key={i}
                          className="inline-flex items-center rounded-full bg-blue-50 px-2.5 py-0.5 text-xs font-medium text-blue-700"
                        >
                          {type}
                        </span>
                      ))}
                    </div>
                  )}
                  
                  {assessment.duration_minutes && (
                    <span className="inline-flex items-center rounded-full bg-gray-100 px-2.5 py-0.5 text-xs font-medium text-gray-700">
                      ⏱ {assessment.duration_minutes} min
                    </span>
                  )}

                  {assessment.job_levels && assessment.job_levels.length > 0 && (
                    <span className="inline-flex items-center rounded-full bg-purple-50 px-2.5 py-0.5 text-xs font-medium text-purple-700">
                      {assessment.job_levels.join(', ')}
                    </span>
                  )}
                </div>

                <div className="mt-4">
                  <a
                    href={assessment.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 text-sm font-medium text-green-600 hover:text-green-700"
                  >
                    View Assessment
                    <svg
                      className="h-4 w-4"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                      />
                    </svg>
                  </a>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
