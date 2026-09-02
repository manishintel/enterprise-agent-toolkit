'use client';

import React, { useState } from 'react';
import { Button, Card, Typography, Alert, Space, Input, Form } from 'antd';
import { LoginOutlined } from '@ant-design/icons';
import { useRouter } from 'next/navigation';
import { useNextAuth } from '../hooks';

const { Title, Text } = Typography;

export interface AuthFormsProps {
  onSuccess?: () => void;
  className?: string;
  onModeChange?: (mode: 'login' | 'register') => void;
}

export const LoginForm: React.FC<AuthFormsProps> = ({ className, onSuccess }) => {
  const { loginWithCredentials, isLoading } = useNextAuth();
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const router = useRouter();
  const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';

  const handleSubmit = async ({ username }: { username: string }) => {
    setError(null);
    setSubmitting(true);
    try {
      await loginWithCredentials(username);
      onSuccess?.();
      router.push(basePath || '/');
    } catch (err: any) {
      setError(err?.message || 'Sign-in failed. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Card className={className} style={{ maxWidth: 400, margin: '0 auto', boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)' }}>
      <div style={{ textAlign: 'center', marginBottom: 24 }}>
        <Title level={3}>Sign In</Title>
        <Text type="secondary">Enter a username to continue</Text>
      </div>

      {error && (
        <Alert
          message="Sign-in failed"
          description={error}
          type="error"
          style={{ marginBottom: 16 }}
          showIcon
          closable
          onClose={() => setError(null)}
        />
      )}

      <Form layout="vertical" onFinish={handleSubmit} disabled={submitting || isLoading}>
        <Form.Item
          label="Username"
          name="username"
          rules={[
            { required: true, message: 'Username is required' },
            {
              pattern: /^[a-zA-Z0-9._-]{1,64}$/,
              message: 'Use letters, digits, dot, hyphen, or underscore (max 64 chars).',
            },
          ]}
        >
          <Input placeholder="your-name" autoFocus autoComplete="username" />
        </Form.Item>

        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
          <Button
            type="primary"
            htmlType="submit"
            size="large"
            block
            icon={<LoginOutlined />}
            loading={submitting || isLoading}
          >
            Sign in
          </Button>
        </Space>
      </Form>
    </Card>
  );
};

export interface AuthModalProps {
  mode: 'login' | 'register';
  onModeChange: (mode: 'login' | 'register') => void;
  onSuccess?: () => void;
  className?: string;
}

export const AuthModal: React.FC<AuthModalProps> = ({
  mode,
  onModeChange,
  onSuccess,
  className,
}) => {
  return (
    <div className={className}>
        <LoginForm onSuccess={onSuccess} onModeChange={onModeChange} />
    </div>
  );
};