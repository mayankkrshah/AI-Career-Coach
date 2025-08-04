'use client';

import React from 'react';
import {
  Box,
  List,
  ListItem,
  ListItemButton,
  IconButton,
  Typography,
  Button,
  TextField,
  Tooltip
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Chat as ChatIcon,
  VpnKey as KeyIcon,
  Settings as SettingsIcon
} from '@mui/icons-material';
import { Session } from '../SessionContext';
import CareerCoachToggle from './CareerCoachToggle';

interface ChatSidebarProps {
  sessions: Session[];
  currentSessionId: string;
  onNewSession: () => void;
  onSwitchSession: (id: string) => void;
  onDeleteSession: (id: string) => void;
  apiKey: string;
  setApiKey: (key: string) => void;
  useCareerCoach: boolean;
  setUseCareerCoach: (value: boolean) => void;
  tavilyApiKey: string;
  setTavilyApiKey: (value: string) => void;
  cohereApiKey: string;
  setCohereApiKey: (value: string) => void;
  useReranking: boolean;
  setUseReranking: (value: boolean) => void;
}

export default function ChatSidebar({
  sessions,
  currentSessionId,
  onNewSession,
  onSwitchSession,
  onDeleteSession,
  apiKey,
  setApiKey,
  useCareerCoach,
  setUseCareerCoach,
  tavilyApiKey,
  setTavilyApiKey,
  cohereApiKey,
  setCohereApiKey,
  useReranking,
  setUseReranking
}: ChatSidebarProps) {
  const [showApiKey, setShowApiKey] = React.useState(false);

  const handleApiKeyChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const key = event.target.value;
    setApiKey(key);
    sessionStorage.setItem('OPENAI_API_KEY', key);
    window.dispatchEvent(new Event('apiKeyChanged'));
  };

  const formatSessionName = (session: Session) => {
    if (session.messages.length === 0) {
      return 'New Chat';
    }
    const firstMessage = session.messages[0];
    return firstMessage.text.slice(0, 30) + (firstMessage.text.length > 30 ? '...' : '');
  };

  const getSessionPreview = (session: Session) => {
    const messageCount = session.messages.length;
    if (messageCount === 0) return 'No messages';
    
    const userMessages = session.messages.filter(m => m.sender === 'user').length;
    const botMessages = session.messages.filter(m => m.sender === 'bot').length;
    
    return `${userMessages} questions, ${botMessages} responses`;
  };

  return (
    <Box
      sx={{
        width: 320,
        height: '100vh',
        bgcolor: '#fafafa',
        borderRight: '1px solid #e0e0e0',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden'
      }}
    >
      <Box sx={{ p: 2, borderBottom: '1px solid #e0e0e0' }}>
        <Typography variant="h6" sx={{ fontWeight: 700, color: '#1976d2', mb: 1 }}>
          AI Career Coach
        </Typography>
        
        <Button
          startIcon={<AddIcon />}
          variant="contained"
          fullWidth
          onClick={onNewSession}
          sx={{ mb: 2 }}
        >
          New Chat
        </Button>

        <Box sx={{ mb: 2 }}>
          <TextField
            label="OpenAI API Key"
            type={showApiKey ? 'text' : 'password'}
            size="small"
            fullWidth
            value={apiKey}
            onChange={handleApiKeyChange}
            placeholder="sk-..."
            error={!useCareerCoach && !apiKey.startsWith('sk-')}
            helperText={
              useCareerCoach 
                ? (!apiKey.startsWith('sk-') 
                   ? 'Optional - for AI-powered analysis' 
                   : 'API key configured - AI features enabled')
                : (!apiKey.startsWith('sk-') 
                   ? 'Enter valid OpenAI API key' 
                   : 'API key configured')
            }
            InputProps={{
              startAdornment: <KeyIcon sx={{ mr: 1, color: '#666' }} />,
              endAdornment: (
                <Tooltip title={showApiKey ? 'Hide key' : 'Show key'}>
                  <IconButton
                    size="small"
                    onClick={() => setShowApiKey(!showApiKey)}
                  >
                    <SettingsIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              )
            }}
          />
        </Box>

        <CareerCoachToggle
          useCareerCoach={useCareerCoach}
          setUseCareerCoach={setUseCareerCoach}
          tavilyApiKey={tavilyApiKey}
          setTavilyApiKey={setTavilyApiKey}
          cohereApiKey={cohereApiKey}
          setCohereApiKey={setCohereApiKey}
          useReranking={useReranking}
          setUseReranking={setUseReranking}
        />
      </Box>

      <Box sx={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <Typography variant="subtitle2" sx={{ p: 2, pb: 1, color: '#666', fontWeight: 600 }}>
          Recent Chats ({sessions.length})
        </Typography>
        
        <Box sx={{ flex: 1, overflowY: 'auto' }}>
          <List sx={{ p: 0 }}>
            {sessions.map((session) => {
              const isActive = session.id === currentSessionId;
              return (
                <ListItem
                  key={session.id}
                  disablePadding
                  sx={{
                    borderLeft: isActive ? '3px solid #1976d2' : '3px solid transparent',
                    bgcolor: isActive ? 'rgba(25, 118, 210, 0.08)' : 'transparent'
                  }}
                >
                  <ListItemButton
                    onClick={() => onSwitchSession(session.id)}
                    sx={{
                      py: 1.5,
                      px: 2,
                      '&:hover': {
                        bgcolor: 'rgba(25, 118, 210, 0.04)'
                      }
                    }}
                  >
                    <ChatIcon sx={{ mr: 1.5, color: isActive ? '#1976d2' : '#666', fontSize: 18 }} />
                    
                    <Box sx={{ flex: 1, minWidth: 0 }}>
                      <Typography
                        variant="body2"
                        sx={{
                          fontWeight: isActive ? 600 : 400,
                          color: isActive ? '#1976d2' : '#333',
                          whiteSpace: 'nowrap',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis'
                        }}
                      >
                        {formatSessionName(session)}
                      </Typography>
                      
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        sx={{
                          display: 'block',
                          whiteSpace: 'nowrap',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis'
                        }}
                      >
                        {getSessionPreview(session)}
                      </Typography>
                    </Box>

                    <Tooltip title="Delete chat">
                      <IconButton
                        size="small"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteSession(session.id);
                        }}
                        sx={{
                          opacity: 0.6,
                          '&:hover': {
                            opacity: 1,
                            color: '#f44336',
                            bgcolor: 'rgba(244, 67, 54, 0.1)'
                          }
                        }}
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </ListItemButton>
                </ListItem>
              );
            })}
          </List>
        </Box>
      </Box>

      <Box sx={{ p: 2, borderTop: '1px solid #e0e0e0', bgcolor: '#f5f5f5' }}>
        <Typography variant="caption" color="text.secondary" align="center" display="block">
          AI Career Coach v1.0
        </Typography>
        <Typography variant="caption" color="text.secondary" align="center" display="block">
          Powered by RAG + LangGraph
        </Typography>
      </Box>
    </Box>
  );
}