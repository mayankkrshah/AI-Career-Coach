'use client';

import React, { useState, useEffect } from 'react';
import { Box, CssBaseline } from '@mui/material';
import { useSessionContext } from './SessionContext';
import ChatSidebar from './components/ChatSidebar';
import MainChatInterface from './components/MainChatInterface';

declare global {
  interface Window {
    env?: {
      NEXT_PUBLIC_OPENAI_API_KEY?: string;
    };
  }
}

export default function ChatPage() {
  const {
    sessions,
    currentSessionId,
    handleSwitchSession,
    handleNewSession,
    handleDeleteSession,
    systemPrompt,
    setSystemPrompt,
    selectedTemplate,
    setSelectedTemplate
  } = useSessionContext();

  // API Keys state
  const [apiKey, setApiKey] = useState('');
  const [tavilyApiKey, setTavilyApiKey] = useState('');
  const [cohereApiKey, setCohereApiKey] = useState('');
  const [langchainApiKey, setLangchainApiKey] = useState('');
  
  // Settings state
  const [useCareerCoach, setUseCareerCoach] = useState(true);
  const [useReranking, setUseReranking] = useState(false);
  const [model, setModel] = useState('gpt-4o-mini');

  // Load API keys and settings from localStorage on mount
  useEffect(() => {
    const getKey = (key: string) => sessionStorage.getItem(key) || '';
    setApiKey(getKey('OPENAI_API_KEY'));
    setTavilyApiKey(getKey('TAVILY_API_KEY'));
    setCohereApiKey(getKey('COHERE_API_KEY'));
    setLangchainApiKey(getKey('LANGCHAIN_API_KEY'));
    const savedCareerCoach = sessionStorage.getItem('USE_CAREER_COACH');
    setUseCareerCoach(savedCareerCoach === null ? true : savedCareerCoach === 'true');
    setUseReranking(sessionStorage.getItem('USE_RERANKING') === 'true');
    
    const onStorage = (e: StorageEvent) => {
      if (e.key === 'OPENAI_API_KEY') setApiKey(getKey('OPENAI_API_KEY'));
      if (e.key === 'TAVILY_API_KEY') setTavilyApiKey(getKey('TAVILY_API_KEY'));
      if (e.key === 'COHERE_API_KEY') setCohereApiKey(getKey('COHERE_API_KEY'));
      if (e.key === 'LANGCHAIN_API_KEY') setLangchainApiKey(getKey('LANGCHAIN_API_KEY'));
      if (e.key === 'USE_CAREER_COACH') setUseCareerCoach(e.newValue === 'true');
      if (e.key === 'USE_RERANKING') setUseReranking(e.newValue === 'true');
    };
    
    const onApiKeyChanged = () => {
      setApiKey(getKey('OPENAI_API_KEY'));
      setTavilyApiKey(getKey('TAVILY_API_KEY'));
      setCohereApiKey(getKey('COHERE_API_KEY'));
      setLangchainApiKey(getKey('LANGCHAIN_API_KEY'));
    };
    
    window.addEventListener('storage', onStorage);
    window.addEventListener('apiKeyChanged', onApiKeyChanged);
    
    return () => {
      window.removeEventListener('storage', onStorage);
      window.removeEventListener('apiKeyChanged', onApiKeyChanged);
    };
  }, []);

  return (
    <>
      <CssBaseline />
      <Box sx={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
        {/* Sidebar */}
        <ChatSidebar
          sessions={sessions}
          currentSessionId={currentSessionId}
          onNewSession={handleNewSession}
          onSwitchSession={handleSwitchSession}
          onDeleteSession={handleDeleteSession}
          apiKey={apiKey}
          setApiKey={setApiKey}
          useCareerCoach={useCareerCoach}
          setUseCareerCoach={setUseCareerCoach}
          tavilyApiKey={tavilyApiKey}
          setTavilyApiKey={setTavilyApiKey}
          cohereApiKey={cohereApiKey}
          setCohereApiKey={setCohereApiKey}
          useReranking={useReranking}
          setUseReranking={setUseReranking}
        />
        
        {/* Main Chat Interface */}
        <MainChatInterface
          apiKey={apiKey}
          useCareerCoach={useCareerCoach}
          tavilyApiKey={tavilyApiKey}
          cohereApiKey={cohereApiKey}
          langchainApiKey={langchainApiKey}
          useReranking={useReranking}
          model={model}
        />
      </Box>
    </>
  );
}