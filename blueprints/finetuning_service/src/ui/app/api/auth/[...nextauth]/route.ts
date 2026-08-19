import NextAuth, { AuthOptions, Session, User } from 'next-auth';
import { JWT } from 'next-auth/jwt';
import CredentialsProvider from 'next-auth/providers/credentials';

const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';
const SESSION_MAX_AGE_SEC = 60 * 60 * 8; // 8 hours

function normalizeUsername(input: unknown): string | null {
  if (typeof input !== 'string') return null;
  const trimmed = input.trim().toLowerCase();
  if (!trimmed) return null;
  // Allow letters, digits, dot, hyphen, underscore. Reject anything else.
  if (!/^[a-z0-9._-]{1,64}$/.test(trimmed)) return null;
  return trimmed;
}

export const authOptions: AuthOptions = {
  providers: [
    CredentialsProvider({
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
          // accessToken doubles as the base64(user_id) fallback the API accepts
          accessToken: Buffer.from(username, 'utf-8').toString('base64'),
        } as User & { accessToken: string };
      },
    }),
  ],
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
        token.provider = 'credentials';
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
