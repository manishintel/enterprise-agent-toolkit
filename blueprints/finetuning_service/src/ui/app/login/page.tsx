'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { signIn } from 'next-auth/react';
import { Typography, Spin } from 'antd';
import { AuthModal, useNextAuth } from '@/src/features/auth';
import { useGlobalState } from '@core/state/globalState';

const { Title } = Typography;

export default function LoginPage() {
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');
  const { isAuthenticated, isLoading } = useNextAuth();
  const { state } = useGlobalState();
  const { theme: { mode: theme } } = state;
  const router = useRouter();
  // Whether the gateway-identity sign-in attempt has finished. Until it has, we
  // show a spinner rather than the username form, so a user behind SSO never
  // sees a prompt they are not supposed to answer.
  const [ssoSettled, setSsoSettled] = useState(false);

  useEffect(() => {
    if (isAuthenticated) {
      router.push('/');
    }
  }, [isAuthenticated, router]);

  /*
   * When an API gateway in front of this app has already authenticated the user
   * (OIDC), it forwards the verified identity as a request header. Try to turn
   * that into a session silently, so SSO users are not asked to log in twice.
   *
   * Deliberately attempted unconditionally instead of behind a NEXT_PUBLIC_
   * build-time flag: those are baked into the bundle at build time, which would
   * mean rebuilding the image just to switch auth modes. If the provider is not
   * registered, or no header is present, this fails harmlessly and the normal
   * form is shown.
   */
  useEffect(() => {
    if (isLoading || isAuthenticated || ssoSettled) return;
    let cancelled = false;
    (async () => {
      try {
        const res = await signIn('oidc-proxy', { redirect: false });
        if (!cancelled && !res?.error) {
          router.push('/');
          return;
        }
      } catch {
        // fall through to the form
      }
      if (!cancelled) setSsoSettled(true);
    })();
    return () => {
      cancelled = true;
    };
  }, [isLoading, isAuthenticated, ssoSettled, router]);

  const handleAuthSuccess = () => {
    router.push('/');
  };

  const handleAuthModeChange = (mode: 'login' | 'register') => {
    setAuthMode(mode);
  };

  if (isAuthenticated) {
    return null; // Will redirect in useEffect
  }

  // The silent gateway-identity sign-in is still in flight. Show a spinner
  // rather than briefly flashing a username form at an SSO user.
  if (!ssoSettled) {
    return (
      <div style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: theme === 'dark' ? '#090B1C' : '#F2F3FF',
      }}>
        <Spin size="large" />
      </div>
    );
  }

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '24px',
      backgroundColor: theme === 'dark' ? '#090B1C' : '#F2F3FF',
    }}>
      <div style={{ width: '100%', maxWidth: '500px' }}>
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <Title level={2} style={{
            color: theme === 'dark' ? '#c9d1d9' : '#3D447F',
            marginBottom: '8px'
          }}>
            Intel AI for Enterprise Finetuning
          </Title>
          <Typography.Text type="secondary">
            Sign in to access your fine-tuning workspace
          </Typography.Text>
        </div>
        <AuthModal
          mode={authMode}
          onModeChange={handleAuthModeChange}
          onSuccess={handleAuthSuccess}
        />
      </div>
    </div>
  );
}