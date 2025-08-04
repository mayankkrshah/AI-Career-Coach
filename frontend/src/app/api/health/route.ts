import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  try {
    // Forward the request to the backend
    const backendResponse = await fetch('http://localhost:9000/api/health', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!backendResponse.ok) {
      throw new Error(`Backend error: ${backendResponse.status}`);
    }

    const data = await backendResponse.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Health check error:', error);
    return NextResponse.json(
      { error: 'Backend unavailable' },
      { status: 503 }
    );
  }
}