export const getOrCreateSessionId = () => {
    const existingSessionId = sessionStorage.getItem('session_id');
    if (existingSessionId) {
      return existingSessionId;
    }
    
    const newSessionId = 'session_' + Math.random().toString(36).substr(2, 9);
    sessionStorage.setItem('session_id', newSessionId);
    return newSessionId;
  };
  