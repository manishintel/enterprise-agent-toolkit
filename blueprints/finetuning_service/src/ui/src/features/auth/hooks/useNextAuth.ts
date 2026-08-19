'use client';
import { useSession, signIn, signOut } from 'next-auth/react';
import { useCallback } from 'react';

export const useNextAuth = () => {
  const { data: session, status } = useSession();

  const loginWithCredentials = useCallback(
    async (username: string) => {
      const res = await signIn('credentials', {
        username,
        redirect: false,
      });
      if (res?.error) throw new Error(res.error);
      return res;
    },
    [],
  );

  const logout = useCallback(async () => {
    await signOut();
  }, []);

  return {
    session,
    user: session?.user || null,
    accessToken: session?.accessToken,
    provider: session?.provider,

    isLoading: status === 'loading',
    isAuthenticated: status === 'authenticated',
    isUnauthenticated: status === 'unauthenticated',

    loginWithCredentials,
    // Backward-compat alias for any lingering imports.
    loginWithKeycloak: loginWithCredentials,
    logout,

    error: session?.error,
  };
};
