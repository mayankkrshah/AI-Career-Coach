export const getApiUrl = () => {
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  // Always use the Next.js API routes which proxy to the backend
  return '/api';
}; 