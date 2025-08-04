'use client';

import React from "react";
import { AppRouterCacheProvider } from '@mui/material-nextjs/v13-appRouter';
import { ThemeProvider } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';
import { Inter } from "next/font/google";
import "./globals.css";
import Script from 'next/script';
import theme from './theme';
import { SessionProvider } from './SessionContext';

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export default function RootLayout(props: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <Script id="env-config" strategy="beforeInteractive">
          {`window.env = ${JSON.stringify({
            NEXT_PUBLIC_OPENAI_API_KEY: process.env.NEXT_PUBLIC_OPENAI_API_KEY
          })};`}
        </Script>
      </head>
      <body className={inter.className} style={{ 
        background: '#f7f8fa',
        minHeight: '100vh', 
        minWidth: '100vw', 
        margin: 0, 
        padding: 0 
      }}>
        <AppRouterCacheProvider>
          <ThemeProvider theme={theme}>
            <SessionProvider>
              <CssBaseline />
              {props.children}
            </SessionProvider>
          </ThemeProvider>
        </AppRouterCacheProvider>
      </body>
    </html>
  );
}