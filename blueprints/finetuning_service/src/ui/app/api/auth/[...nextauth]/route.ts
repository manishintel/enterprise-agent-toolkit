import NextAuth, { AuthOptions, Session, User } from 'next-auth';
import { JWT } from 'next-auth/jwt';
import CredentialsProvider from 'next-auth/providers/credentials';

const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';
const SESSION_MAX_AGE_SEC = 60 * 60 * 8; // 8 hours

/*
 * Two mutually exclusive ways to establish a session:
 *
 *  - 'oidc-proxy'  : identity comes from a header injected by the API gateway
 *                    after it completed an OIDC login. Used when the gateway
 *                    enforces authentication in front of this app.
 *  - 'credentials' : the legacy username-only prompt. No password is checked,
 *                    so it is only acceptable when nothing else is available.
 *
 * When OIDC_PROXY_AUTH_ENABLED is set, the credentials provider is removed
 * entirely rather than merely hidden — leaving it registered would keep a
 * passwordless path to a session even though the gateway is enforcing OIDC.
 */
const PROXY_AUTH_ENABLED = process.env.OIDC_PROXY_AUTH_ENABLED === 'true';

// Header carrying the gateway-verified identity. APISIX's openid-connect
// plugin populates X-Userinfo with base64(JSON of the userinfo claims).
const USERINFO_HEADER = (process.env.OIDC_USERINFO_HEADER || 'x-userinfo').toLowerCase();
// Fallback header for gateways that forward a bare username instead.
const USER_HEADER = (process.env.OIDC_USER_HEADER || 'x-forwarded-user').toLowerCase();
// Claim precedence, configurable so a differently-configured IdP does not
// require rebuilding this image.
const USERNAME_CLAIMS = (process.env.OIDC_USERNAME_CLAIMS || 'preferred_username,email,sub')
  .split(',')
  .map((c) => c.trim())
  .filter(Boolean);

function normalizeUsername(input: unknown): string | null {
  if (typeof input !== 'string') return null;
  const trimmed = input.trim().toLowerCase();
  if (!trimmed) return null;
  // Allow letters, digits, dot, hyphen, underscore, plus @ for email-shaped
  // claims. Reject anything else.
  if (!/^[a-z0-9._@-]{1,128}$/.test(trimmed)) return null;
  return trimmed;
}

/** Decode a base64 / base64url payload into an object, or null. */
function decodeUserinfo(raw: string): Record<string, unknown> | null {
  try {
    const b64 = raw.replace(/-/g, '+').replace(/_/g, '/');
    const padded = b64 + '='.repeat((4 - (b64.length % 4)) % 4);
    const json = Buffer.from(padded, 'base64').toString('utf-8');
    const parsed = JSON.parse(json);
    return parsed && typeof parsed === 'object' ? parsed : null;
  } catch {
    return null;
  }
}

/** Pull a username out of the gateway-injected headers. */
function usernameFromHeaders(headers: Record<string, unknown> | undefined): string | null {
  if (!headers) return null;
  const get = (name: string): string | undefined => {
    const v = (headers as Record<string, string | string[] | undefined>)[name];
    return Array.isArray(v) ? v[0] : v;
  };

  const rawUserinfo = get(USERINFO_HEADER);
  if (rawUserinfo) {
    const claims = decodeUserinfo(rawUserinfo);
    if (claims) {
      for (const claim of USERNAME_CLAIMS) {
        const candidate = normalizeUsername(claims[claim]);
        if (candidate) return candidate;
      }
    }
  }

  return normalizeUsername(get(USER_HEADER));
}

const oidcProxyProvider = CredentialsProvider({
  id: 'oidc-proxy',
  name: 'Single Sign-On',
  // No user-supplied fields: the identity is taken from the request headers,
  // never from anything the browser can type.
  credentials: {},
  async authorize(credentials, req) {
    void credentials; // identity comes from headers only, never from input
    const username = usernameFromHeaders(
      req?.headers as unknown as Record<string, unknown> | undefined,
    );
    if (!username) return null;
    return {
      id: username,
      name: username,
      email: username.includes('@') ? username : `${username}@local`,
      accessToken: Buffer.from(username, 'utf-8').toString('base64'),
    } as User & { accessToken: string };
  },
});

const legacyCredentialsProvider = CredentialsProvider({
  id: 'credentials',
  name: 'Username',
  credentials: {
    username: { label: 'Username', type: 'text', placeholder: 'your-name' },
  },
  async authorize(credentials) {
    const username = normalizeUsername(credentials?.username);
    if (!username) return null;
    return {
      id: username,
      name: username,
      email: `${username}@local`,
      accessToken: Buffer.from(username, 'utf-8').toString('base64'),
    } as User & { accessToken: string };
  },
});

export const authOptions: AuthOptions = {
  providers: PROXY_AUTH_ENABLED ? [oidcProxyProvider] : [legacyCredentialsProvider],
  session: {
    strategy: 'jwt',
    maxAge: SESSION_MAX_AGE_SEC,
  },
  pages: {
    signIn: `${basePath}/login`,
    error: `${basePath}/login`,
  },
  callbacks: {
    async jwt({ token, user }: { token: JWT; user?: User }) {
      if (user) {
        token.provider = PROXY_AUTH_ENABLED ? 'oidc-proxy' : 'credentials';
        token.accessToken = (user as User & { accessToken?: string }).accessToken;
        token.user = {
          id: user.id ?? '',
          email: user.email ?? null,
          name: user.name ?? null,
        };
      }
      return token;
    },
    async session({ session, token }: { session: Session; token: JWT }) {
      if (token) {
        session.user = (token.user as Session['user']) ?? session.user;
        session.accessToken = token.accessToken as string | undefined;
        session.provider = token.provider as string | undefined;
      }
      return session;
    },
  },
  debug: process.env.NODE_ENV === 'development',
};

const handler = NextAuth(authOptions);

export { handler as GET, handler as POST };
