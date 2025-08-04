'use client';

import React from 'react';
import {
  Snackbar,
  Alert,
  Typography,
  Box,
  Chip
} from '@mui/material';
import {
  Speed as SpeedIcon,
  Storage as DatabaseIcon,
  Language as InternetIcon,
  School as AcademicIcon,
  Psychology as AIIcon
} from '@mui/icons-material';

interface ToolUsageToastProps {
  open: boolean;
  onClose: () => void;
  toolsUsed: string[];
  primarySource: string;
  processingTime: number;
}

export default function ToolUsageToast({ 
  open, 
  onClose, 
  toolsUsed, 
  primarySource, 
  processingTime 
}: ToolUsageToastProps) {
  const getSourceIcon = (source: string) => {
    if (source.includes('RAG') || source.includes('Database')) {
      return <DatabaseIcon fontSize="small" />;
    }
    if (source.includes('Internet') || source.includes('Search')) {
      return <InternetIcon fontSize="small" />;
    }
    if (source.includes('Academic') || source.includes('ArXiv') || source.includes('Papers')) {
      return <AcademicIcon fontSize="small" />;
    }
    if (source.includes('AI Assistant') || source.includes('GPT')) {
      return <AIIcon fontSize="small" />;
    }
    return <AIIcon fontSize="small" />;
  };

  const getSourceColor = (source: string) => {
    if (source.includes('RAG') || source.includes('Database')) {
      return 'success';
    }
    if (source.includes('Internet') || source.includes('Search')) {
      return 'info';
    }
    if (source.includes('Academic') || source.includes('ArXiv') || source.includes('Papers')) {
      return 'warning';
    }
    if (source.includes('AI Assistant') || source.includes('GPT')) {
      return 'primary';
    }
    return 'default';
  };

  return (
    <Snackbar 
      open={open} 
      autoHideDuration={4000} 
      onClose={onClose}
      anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
    >
      <Alert 
        severity="info" 
        variant="filled"
        onClose={onClose}
        sx={{ 
          minWidth: 300,
          bgcolor: '#1976d2',
          color: '#fff'
        }}
      >
        <Box>
          <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
            Response from: {primarySource}
          </Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            {getSourceIcon(primarySource)}
            <Chip 
              label={primarySource}
              size="small"
              color={getSourceColor(primarySource) as any}
              variant="outlined"
              sx={{ 
                color: '#fff',
                borderColor: '#fff',
                '& .MuiChip-label': { color: '#fff' }
              }}
            />
          </Box>
          
          {processingTime > 0 && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <SpeedIcon fontSize="small" />
              <Typography variant="caption">
                Processed in {processingTime.toFixed(1)}s
              </Typography>
            </Box>
          )}
        </Box>
      </Alert>
    </Snackbar>
  );
}