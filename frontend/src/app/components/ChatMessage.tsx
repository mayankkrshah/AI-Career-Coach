'use client';

import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  IconButton,
  Avatar,
  Tooltip
} from '@mui/material';
import {
  Person as UserIcon,
  SmartToy as BotIcon,
  Delete as DeleteIcon,
  Error as ErrorIcon
} from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import { Message } from '../SessionContext';

interface ChatMessageProps {
  message: Message;
  onDelete: () => void;
}

export default function ChatMessage({ message, onDelete }: ChatMessageProps) {
  const [isHovered, setIsHovered] = useState(false);
  const isUser = message.sender === 'user';

  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        mb: 2,
        alignItems: 'flex-start',
        gap: 1
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {!isUser && (
        <Avatar
          sx={{
            bgcolor: message.isError ? '#f44336' : '#1976d2',
            width: 32,
            height: 32,
            mt: 0.5
          }}
        >
          {message.isError ? <ErrorIcon fontSize="small" /> : <BotIcon fontSize="small" />}
        </Avatar>
      )}

      <Box sx={{ maxWidth: '70%', minWidth: '200px' }}>
        <Paper
          elevation={1}
          sx={{
            p: 2,
            bgcolor: isUser 
              ? '#1976d2' 
              : message.isError 
                ? '#ffebee' 
                : '#f5f5f5',
            color: isUser ? '#fff' : message.isError ? '#c62828' : '#333',
            borderRadius: 2,
            position: 'relative',
            border: message.isError ? '1px solid #f44336' : 'none'
          }}
        >
          {message.isError && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <ErrorIcon fontSize="small" color="error" />
              <Typography variant="caption" color="error" fontWeight={600}>
                Error
              </Typography>
            </Box>
          )}

          <Typography 
            variant="body1" 
            component="div"
            sx={{ 
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              '& p': { margin: 0, mb: 1 },
              '& p:last-child': { mb: 0 },
              '& code': {
                bgcolor: isUser ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.1)',
                px: 0.5,
                py: 0.25,
                borderRadius: 0.5,
                fontSize: '0.875rem'
              },
              '& pre': {
                bgcolor: isUser ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)',
                p: 1,
                borderRadius: 1,
                overflow: 'auto',
                fontSize: '0.875rem'
              }
            }}
          >
            {isUser ? (
              message.text
            ) : (
              <ReactMarkdown>{message.text}</ReactMarkdown>
            )}
          </Typography>

          {isHovered && (
            <Tooltip title="Delete message">
              <IconButton
                size="small"
                onClick={onDelete}
                sx={{
                  position: 'absolute',
                  top: -8,
                  right: -8,
                  bgcolor: '#f44336',
                  color: '#fff',
                  width: 20,
                  height: 20,
                  '&:hover': {
                    bgcolor: '#d32f2f'
                  }
                }}
              >
                <DeleteIcon sx={{ fontSize: 12 }} />
              </IconButton>
            </Tooltip>
          )}
        </Paper>

        <Typography 
          variant="caption" 
          color="text.secondary" 
          sx={{ 
            display: 'block', 
            mt: 0.5,
            textAlign: isUser ? 'right' : 'left'
          }}
        >
          {isUser ? 'You' : message.isError ? 'Error' : 'AI Assistant'}
        </Typography>
      </Box>

      {isUser && (
        <Avatar
          sx={{
            bgcolor: '#4caf50',
            width: 32,
            height: 32,
            mt: 0.5
          }}
        >
          <UserIcon fontSize="small" />
        </Avatar>
      )}
    </Box>
  );
}