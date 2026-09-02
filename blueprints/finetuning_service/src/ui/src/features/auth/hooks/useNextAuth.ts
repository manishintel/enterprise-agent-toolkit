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

  /*
   * Signing out has to tear down TWO sessions, not one.
   *
   * signOut() only clears this app's NextAuth cookie. When an API gateway in
   * front of us performed the OIDC login, its own session cookie survives — so
   * the very next request still arrives with a verified identity header, the
   * login page silently re-authenticates from it, and the user lands straight
   * back on the dashboard having apparently never logged out.
   *
   * So after clearing the local session, hand the browser to the gateway's
   * logout path. The gateway destroys its session and redirects on to the
   * provider's end-session endpoint, which also ends the SSO session — without
   * that last part, the next login would skip the password prompt.
   *
   * Keyed off session.provider rather than a build-time flag so a deployment
   * without gateway auth still just returns to the local login page.
   */
  const logout = useCallback(async () => {
    const viaGateway = session?.provider === 'oidc-proxy';
    const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';
    const target = viaGateway ? `${basePath}/logout` : `${basePath}/login`;
    /*
     * signOut() must never be able to prevent the gateway logout.
     *
     * If it throws (its own fetches go through the gateway and can fail), the
     * navigation below would be skipped and NextAuth's client would instead send
     * the browser to its built-in error route under /api/auth/error — a path the
     * gateway answers with a bare 401, so the user sees "401 Authorization
     * Required" instead of being logged out. Swallow the failure and always hand
     * over to the gateway, which destroys its session and the IdP's.
     */
    /*
     * Hand the redirect to NextAuth via callbackUrl instead of clearing the
     * session and navigating ourselves.
     *
     * With redirect:false + window.location, TWO navigations race: ours to the
     * gateway logout, and the app's own client-side routing to /login the moment
     * SessionProvider notices the session vanish. That second page load fires
     * /api/auth/providers, /api/auth/session and further /ui requests, and every
     * unauthenticated one of those makes the gateway start a fresh authorization
     * flow, rewriting the CSRF state behind the login page the user is already
     * looking at. Submitting that form then fails with a state mismatch (500).
     * One navigation chain, driven by NextAuth, avoids the whole race.
     */
    try {
      await signOut({ callbackUrl: target });
    } catch {
      // If NextAuth could not navigate, force it: the gateway logout is what
      // actually ends the session.
      window.location.href = target;
    }
  }, [session]);

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
