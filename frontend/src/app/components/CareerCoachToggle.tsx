'use client';

import React from 'react';
import {
  Box,
  Switch,
  FormControlLabel,
  Typography,
  Chip,
  Collapse,
  TextField,
  Grid,
  Tooltip,
  IconButton
} from '@mui/material';
import {
  SmartToy as AIIcon,
  Settings as SettingsIcon,
  Info as InfoIcon
} from '@mui/icons-material';

interface CareerCoachToggleProps {
  useCareerCoach: boolean;
  setUseCareerCoach: (value: boolean) => void;
  tavilyApiKey: string;
  setTavilyApiKey: (value: string) => void;
  cohereApiKey: string;
  setCohereApiKey: (value: string) => void;
  useReranking: boolean;
  setUseReranking: (value: boolean) => void;
}

export default function CareerCoachToggle({
  useCareerCoach,
  setUseCareerCoach,
  tavilyApiKey,
  setTavilyApiKey,
  cohereApiKey,
  setCohereApiKey,
  useReranking,
  setUseReranking
}: CareerCoachToggleProps) {
  const [showAdvanced, setShowAdvanced] = React.useState(false);

  const handleCareerCoachToggle = (event: React.ChangeEvent<HTMLInputElement>) => {
    const enabled = event.target.checked;
    setUseCareerCoach(enabled);
    sessionStorage.setItem('USE_CAREER_COACH', enabled.toString());
  };

  const handleTavilyKeyChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const key = event.target.value;
    setTavilyApiKey(key);
    sessionStorage.setItem('TAVILY_API_KEY', key);
  };

  const handleCohereKeyChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const key = event.target.value;
    setCohereApiKey(key);
    sessionStorage.setItem('COHERE_API_KEY', key);
  };

  const handleRerankingToggle = (event: React.ChangeEvent<HTMLInputElement>) => {
    const enabled = event.target.checked;
    setUseReranking(enabled);
    sessionStorage.setItem('USE_RERANKING', enabled.toString());
  };

  return (
    <Box sx={{ p: 2, bgcolor: 'rgba(0, 0, 0, 0.02)', borderRadius: 2, mb: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
        <FormControlLabel
          control={
            <Switch
              checked={useCareerCoach}
              onChange={handleCareerCoachToggle}
              color="primary"
            />
          }
          label={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <AIIcon color={useCareerCoach ? 'primary' : 'disabled'} />
              <Typography variant="body2" fontWeight={600}>
                AI Career Coach Mode
              </Typography>
            </Box>
          }
        />
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Tooltip title="Advanced career coaching with job market data, web search, and academic research">
            <IconButton size="small">
              <InfoIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          
          <IconButton
            size="small"
            onClick={() => setShowAdvanced(!showAdvanced)}
            sx={{ transform: showAdvanced ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }}
          >
            <SettingsIcon fontSize="small" />
          </IconButton>
        </Box>
      </Box>

      {useCareerCoach && (
        <Box sx={{ mb: 1 }}>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
            Enhanced mode with access to:
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
            <Chip label="Job Market Database" size="small" color="success" variant="outlined" />
            <Chip label="Web Search" size="small" color="info" variant="outlined" />
            <Chip label="Academic Papers" size="small" color="warning" variant="outlined" />
          </Box>
        </Box>
      )}

      <Collapse in={showAdvanced && useCareerCoach}>
        <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid rgba(0, 0, 0, 0.1)' }}>
          <Typography variant="subtitle2" gutterBottom>
            Optional Enhancements
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <TextField
                label="Tavily API Key (Web Search)"
                type="password"
                size="small"
                fullWidth
                value={tavilyApiKey}
                onChange={handleTavilyKeyChange}
                placeholder="tvly-..."
                helperText="Enable real-time web search for current trends"
              />
            </Grid>
            
            <Grid item xs={12}>
              <TextField
                label="Cohere API Key (Advanced Ranking)"
                type="password"
                size="small"
                fullWidth
                value={cohereApiKey}
                onChange={handleCohereKeyChange}
                placeholder="co-..."
                helperText="Improve search result relevance with AI reranking"
              />
            </Grid>
            
            {cohereApiKey && (
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={useReranking}
                      onChange={handleRerankingToggle}
                      size="small"
                    />
                  }
                  label="Enable Cohere Reranking"
                />
              </Grid>
            )}
          </Grid>
        </Box>
      </Collapse>
    </Box>
  );
}