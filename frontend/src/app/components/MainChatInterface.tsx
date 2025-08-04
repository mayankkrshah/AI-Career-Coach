'use client';

import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  TextField,
  IconButton,
  Typography,
  Paper,
  CircularProgress,
  Alert,
  Snackbar
} from '@mui/material';
import {
  Send as SendIcon,
  AutoMode as AIIcon
} from '@mui/icons-material';
import { useSessionContext } from '../SessionContext';
import { getApiUrl } from '../../utils/config';
import ChatMessage from './ChatMessage';
import ToolUsageToast from './ToolUsageToast';

interface MainChatInterfaceProps {
  apiKey: string;
  useCareerCoach: boolean;
  tavilyApiKey: string;
  cohereApiKey: string;
  langchainApiKey: string;
  useReranking: boolean;
  model: string;
}

export default function MainChatInterface({
  apiKey,
  useCareerCoach,
  tavilyApiKey,
  cohereApiKey,
  langchainApiKey,
  useReranking,
  model
}: MainChatInterfaceProps) {
  const {
    sessions,
    currentSessionId,
    addMessageToCurrentSession,
    updateMessageInCurrentSession,
    deleteMessage
  } = useSessionContext();

  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showToast, setShowToast] = useState(false);
  const [toastData, setToastData] = useState<{
    toolsUsed: string[];
    primarySource: string;
    processingTime: number;
  }>({ toolsUsed: [], primarySource: '', processingTime: 0 });

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const currentSession = sessions.find(s => s.id === currentSessionId);
  const messages = currentSession?.messages || [];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;
    
    // For career coach mode, allow sending without API key (RAG will work)
    if (!useCareerCoach && !apiKey.startsWith('sk-')) return;

    const userMessage = {
      id: Date.now().toString(),
      text: inputMessage,
      sender: 'user' as const
    };

    addMessageToCurrentSession(userMessage);
    setInputMessage('');
    setIsLoading(true);

    const botMessageId = (Date.now() + 1).toString();
    const botMessage = {
      id: botMessageId,
      text: 'Thinking...',
      sender: 'bot' as const
    };
    addMessageToCurrentSession(botMessage);

    try {
      const endpoint = useCareerCoach ? '/career-coach' : '/chat';
      const requestBody = useCareerCoach ? {
        user_message: inputMessage,
        api_key: apiKey,
        ...(tavilyApiKey && { tavily_api_key: tavilyApiKey }),
        ...(cohereApiKey && { cohere_api_key: cohereApiKey }),
        ...(langchainApiKey && { langchain_api_key: langchainApiKey }),
        use_reranking: useReranking
      } : {
        user_message: inputMessage,
        api_key: apiKey,
        system_prompt: 'You are a helpful AI assistant.'
      };

      const response = await fetch(`${getApiUrl()}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      updateMessageInCurrentSession(botMessageId, {
        text: data.response || 'No response received',
      });

      if (data.tools_used && data.tools_used.length > 0) {
        setToastData({
          toolsUsed: data.tools_used,
          primarySource: data.primary_source || 'AI Assistant',
          processingTime: data.processing_time || 0
        });
        setShowToast(true);
      }

    } catch (error) {
      console.error('Error sending message:', error);
      updateMessageInCurrentSession(botMessageId, {
        text: 'Sorry, there was an error processing your message. Please check your API key and try again.',
        isError: true
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <Box sx={{ p: 2, borderBottom: '1px solid #e0e0e0', bgcolor: '#fafafa' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <AIIcon color={useCareerCoach ? 'primary' : 'disabled'} />
          <Typography variant="h6" sx={{ fontWeight: 700, color: '#1976d2' }}>
            {useCareerCoach ? 'AI Career Coach' : 'General Chat'}
          </Typography>
          {useCareerCoach && (
            <Typography variant="caption" color="text.secondary">
              Enhanced with RAG + Multi-Agent
            </Typography>
          )}
        </Box>
        
        {!apiKey.startsWith('sk-') && !useCareerCoach && (
          <Alert severity="warning" sx={{ mt: 1 }}>
            Please enter a valid OpenAI API key to start chatting
          </Alert>
        )}
        {!apiKey.startsWith('sk-') && useCareerCoach && (
          <Alert severity="info" sx={{ mt: 1 }}>
            You can ask career questions using our job database. For AI-powered analysis, add an OpenAI API key.
          </Alert>
        )}
      </Box>

      <Box sx={{ flex: 1, overflowY: 'auto', p: 2 }}>
        {messages.length === 0 ? (
          <Box sx={{ 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center', 
            justifyContent: 'center', 
            height: '100%',
            textAlign: 'center',
            gap: 2
          }}>
            <AIIcon sx={{ fontSize: 48, color: '#1976d2', opacity: 0.5 }} />
            <Typography variant="h5" color="text.secondary">
              {useCareerCoach ? 'Welcome to AI Career Coach' : 'Start a Conversation'}
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 400 }}>
              {useCareerCoach 
                ? 'Ask me about job opportunities, career advice, industry trends, or skill development. I have access to job market data and can search the web for the latest information.'
                : 'I\'m here to help with any questions or tasks you have. What would you like to discuss?'
              }
            </Typography>
          </Box>
        ) : (
          <>
            {messages.map((message) => (
              <ChatMessage
                key={message.id}
                message={message}
                onDelete={() => deleteMessage(message.id!)}
              />
            ))}
            <div ref={messagesEndRef} />
          </>
        )}
      </Box>

      <Box sx={{ p: 2, borderTop: '1px solid #e0e0e0', bgcolor: '#fafafa' }}>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-end' }}>
          <TextField
            multiline
            maxRows={4}
            fullWidth
            variant="outlined"
            placeholder={useCareerCoach ? "Ask about jobs, career advice, or industry trends..." : "Type your message..."}
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={(!useCareerCoach && !apiKey.startsWith('sk-')) || isLoading}
            sx={{
              '& .MuiOutlinedInput-root': {
                borderRadius: 2,
                bgcolor: '#fff'
              }
            }}
          />
          <IconButton
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || (!useCareerCoach && !apiKey.startsWith('sk-')) || isLoading}
            color="primary"
            sx={{ 
              p: 1.5,
              bgcolor: '#1976d2',
              color: '#fff',
              '&:hover': { bgcolor: '#1565c0' },
              '&:disabled': { bgcolor: '#e0e0e0', color: '#bdbdbd' }
            }}
          >
            {isLoading ? <CircularProgress size={20} color="inherit" /> : <SendIcon />}
          </IconButton>
        </Box>
      </Box>

      <ToolUsageToast
        open={showToast}
        onClose={() => setShowToast(false)}
        toolsUsed={toastData.toolsUsed}
        primarySource={toastData.primarySource}
        processingTime={toastData.processingTime}
      />
    </Box>
  );
}