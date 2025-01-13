#ifdef DEVELOPMENT
#include "./local.h"
#endif
// AtCoder用.
#ifdef ATCODER
// boost.
#if __has_include(<boost/multiprecision/cpp_int.hpp>)
#include <boost/multiprecision/cpp_int.hpp>
using namespace boost::multiprecision;
#endif
// atcoder.
#if __has_include(<atcoder/all>)
#include <atcoder/all>
using namespace atcoder;
using mint = modint998244353;
using MINT = modint1000000007;
#endif
#endif
// オンラインジャッジ用.
#if defined(ONLINE_JUDGE) || defined(EVAL)
// bits.
#if __has_include(<bits/stdc++.h>)
#include <bits/stdc++.h>
using namespace std;
#endif
// ヒューリスティック.
#include <chrono>
#include <random>
using namespace chrono;
#endif

// テンプレート.
#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#define rep(i, n) for (ll i = 0; i < (n); ++i)
#define REP(i, cc, n) for (ll i = cc; i < (n); ++i)
#define rep1(i, n) for (ll i = 1; i <= (n); ++i)
#define rrep(i, n) for (ll i = n; i > 0; --i)
#define bitrep(i, n) for (ll i = 0; i < (1 << n); ++i)
#define all(a) (a).begin(), (a).end()
#define yesNo(b) ((b) ? "Yes" : "No")
using ll = long long;
using ull = unsigned long long;
using lll = __int128_t;
using ld = long double;
string alphabet = "abcdefghijklmnopqrstuvwxyz";
string ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
constexpr double pi = 3.141592653589793;
constexpr ll smallMOD = 998244353;
constexpr ll bigMOD = 1000000007;
constexpr ll INF = 1LL << 60;
constexpr ll dx[] = {1, 0, -1, 0, 1, -1, -1, 1};
constexpr ll dy[] = {0, 1, 0, -1, 1, 1, -1, -1};

// init.
struct Init
{
    Init()
    {
        ios::sync_with_stdio(0);
        cin.tie(0);
        cout << fixed << setprecision(15);
    }
} init;
template <typename T>
ostream &operator<<(ostream &os, const vector<T> &vec)
{
    os << "[";
    rep(i, vec.size())
    {
        os << vec[i];
        if (i != vec.size() - 1)
            os << ", ";
    }
    os << "]";
    return os;
}
template <typename T>
ostream &operator<<(ostream &os, const vector<vector<T>> &vec)
{
    os << "[";
    rep(i, vec.size())
    {
        os << vec[i];
        if (i != vec.size() - 1)
            os << ", ";
    }
    os << "]";
    return os;
}
template <typename T, typename U>
ostream &operator<<(ostream &os, const pair<T, U> &pair_var)
{
    os << "(" << pair_var.first << ", " << pair_var.second << ")";
    return os;
}
template <typename T>
ostream &operator<<(ostream &os, const set<T> &st)
{
    os << "{";
    for (auto itr = st.begin(); itr != st.end(); ++itr)
    {
        os << *itr;
        if (next(itr) != st.end())
            os << ", ";
    }
    os << "}";
    return os;
}
template <typename T>
ostream &operator<<(ostream &os, const set<T, greater<T>> &st)
{
    os << "{";
    for (auto itr = st.begin(); itr != st.end(); ++itr)
    {
        os << *itr;
        if (next(itr) != st.end())
            os << ", ";
    }
    os << "}";
    return os;
}
template <typename T>
ostream &operator<<(ostream &os, const multiset<T> &st)
{
    os << "{";
    for (auto itr = st.begin(); itr != st.end(); ++itr)
    {
        os << *itr;
        if (next(itr) != st.end())
            os << ", ";
    }
    os << "}";
    return os;
}
template <typename T>
ostream &operator<<(ostream &os, const multiset<T, greater<T>> &st)
{
    os << "{";
    for (auto itr = st.begin(); itr != st.end(); ++itr)
    {
        os << *itr;
        if (next(itr) != st.end())
            os << ", ";
    }
    os << "}";
    return os;
}
template <typename T>
ostream &operator<<(ostream &os, const queue<T> &que)
{
    queue<T> que2 = que;
    os << "{";
    while (!que2.empty())
    {
        os << que2.front();
        que2.pop();
        if (!que2.empty())
            os << ", ";
    }
    os << "}";
    return os;
}
template <typename T>
ostream &operator<<(ostream &os, const deque<T> &deq)
{
    os << "{";
    rep(i, deq.size())
    {
        os << deq[i];
        if (i != deq.size() - 1)
            os << ", ";
    }
    os << "}";
    return os;
}
template <typename T, typename U>
ostream &operator<<(ostream &os, const map<T, U> &mp)
{
    os << "{";
    for (auto itr = mp.begin(); itr != mp.end(); ++itr)
    {
        os << *itr;
        if (next(itr) != mp.end())
            os << ", ";
    }
    os << "}";
    return os;
}
template <typename T>
ostream &operator<<(ostream &os, const stack<T> &stk)
{
    stack<T> stk2 = stk;
    os << "{";
    while (!stk2.empty())
    {
        os << stk2.top();
        stk2.pop();
        if (!stk2.empty())
            os << ", ";
    }
    os << "}";
    return os;
}
template <typename T>
ostream &operator<<(ostream &os, const priority_queue<T> &que)
{
    priority_queue<T> que2 = que;
    os << "{";
    while (!que2.empty())
    {
        os << que2.top();
        que2.pop();
        if (!que2.empty())
            os << ", ";
    }
    os << "}";
    return os;
}
template <class T>
bool chmax(T &a, T b)
{
    if (a < b)
    {
        a = b;
        return 1;
    }
    return 0;
}
template <class T>
bool chmin(T &a, T b)
{
    if (b < a)
    {
        a = b;
        return 1;
    }
    return 0;
}
template <class T>
size_t HashCombine(const size_t seed, const T &v)
{
    return seed ^ (std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}
template <class T, class S>
struct std::hash<std::pair<T, S>>
{
    size_t operator()(const std::pair<T, S> &keyval) const noexcept
    {
        return HashCombine(std::hash<T>()(keyval.first), keyval.second);
    }
};
template <class T>
struct std::hash<std::vector<T>>
{
    size_t operator()(const std::vector<T> &keyval) const noexcept
    {
        size_t s = 0;
        for (auto &&v : keyval)
            s = HashCombine(s, v);
        return s;
    }
};
template <int N>
struct HashTupleCore
{
    template <class Tuple>
    size_t operator()(const Tuple &keyval) const noexcept
    {
        size_t s = HashTupleCore<N - 1>()(keyval);
        return HashCombine(s, std::get<N - 1>(keyval));
    }
};
template <>
struct HashTupleCore<0>
{
    template <class Tuple>
    size_t operator()(const Tuple &keyval) const noexcept { return 0; }
};
template <class... Args>
struct std::hash<std::tuple<Args...>>
{
    size_t operator()(const tuple<Args...> &keyval) const noexcept
    {
        return HashTupleCore<tuple_size<tuple<Args...>>::value>()(keyval);
    }
};

// 約数列挙→O(√N).
vector<ll> all_divisors(ll N)
{
    vector<ll> res;
    for (ll i = 1; i * i <= N; ++i)
    {
        if (N % i != 0)
            continue;
        res.push_back(i);
        if (N / i != i)
            res.push_back(N / i);
    }
    sort(res.begin(), res.end());
    return res;
}

// 素因数分解→O(√N).
vector<pair<ll, ll>> prime_factorize(ll N)
{
    vector<pair<ll, ll>> res;
    for (ll p = 2; p * p <= N; ++p)
    {
        if (N % p != 0)
            continue;
        ll e = 0;
        while (N % p == 0)
        {
            ++e;
            N /= p;
        }
        res.emplace_back(p, e);
    }
    if (N != 1)
    {
        res.emplace_back(N, 1);
    }
    return res;
}

// 繰り返し二乗法→O(logY).
template <class T>
T pow_mod(T A, T N, T M)
{
    T res = 1 % M;
    A %= M;
    while (N)
    {
        if (N & 1)
            res = (res * A) % M;
        A = (A * A) % M;
        N >>= 1;
    }
    return res;
}

// 回文判定→O(N).
bool isPalindrome(string str)
{
    ll low = 0;
    ll high = str.length() - 1;
    while (low < high)
    {
        if (str[low] != str[high])
        {
            return false;
        }
        low++;
        high--;
    }
    return true;
}
map<char, ll> alphabetHash = {
    {'a', 2},
    {'b', 3},
    {'c', 5},
    {'d', 7},
    {'e', 11},
    {'f', 13},
    {'g', 17},
    {'h', 19},
    {'i', 23},
    {'j', 29},
    {'k', 31},
    {'l', 37},
    {'m', 41},
    {'n', 43},
    {'o', 47},
    {'p', 53},
    {'q', 59},
    {'r', 61},
    {'s', 67},
    {'t', 71},
    {'u', 73},
    {'v', 79},
    {'w', 83},
    {'x', 89},
    {'y', 97},
    {'z', 101},
};
// 文字列の数字に対するmod計算→O(|S|).
ll string_mod(string s, ll mod)
{
    ll rest = 0;
    for (char c : s)
    {
        ll v = c - '0';
        rest = (rest * 10 + v) % mod;
    }
    return rest;
}

// 二次元累積和.
struct CumulativeSum2D
{
    vector<vector<ll>> data;
    CumulativeSum2D(vector<vector<ll>> &source)
    {
        ll h = source.size();
        ll w = 0;
        if (h != 0)
            w = source[0].size();
        data.resize(h + 1, vector<ll>(w + 1, 0));
        rep(i, h)
        {
            rep(j, w)
            {
                data[i + 1][j + 1] = source[i][j];
            }
        }
        rep(i, h + 1)
        {
            rep(j, w)
            {
                data[i][j + 1] += data[i][j];
            }
        }
        rep(i, h)
        {
            rep(j, w + 1)
            {
                data[i + 1][j] += data[i][j];
            }
        }
    }
    // (x1,y1)から(x2,y2)までの矩形和→O(1).
    ll query(ll x1, ll y1, ll x2, ll y2)
    {
        return data[x2][y2] - data[x1][y2] - data[x2][y1] + data[x1][y1];
    }
};

// UnionFind.
struct UnionFind
{
    vector<ll> parent;
    vector<ll> parentSize;
    UnionFind(ll N)
    {
        rep(i, N + 1)
        {
            parent.push_back(i);
            parentSize.push_back(1);
        }
    }
    ll root(ll x)
    {
        if (parent[x] == x)
            return x;
        parent[x] = root(parent[x]);
        return parent[x];
    }
    bool unite(ll x, ll y)
    {
        x = root(x);
        y = root(y);
        if (x == y)
            return false;
        if (parentSize[x] > parentSize[y])
            swap(x, y);
        parent[x] = y;
        parentSize[y] += parentSize[x];
        return true;
    }
    bool same(ll x, ll y)
    {
        return root(x) == root(y);
    }
    ll size(ll x)
    {
        return parentSize[root(x)];
    }
};
struct RollBackUnionFind
{
    vector<ll> parent;
    stack<pair<ll, ll>> history;
    RollBackUnionFind(ll N)
    {
        parent.assign(N, -1);
    }
    ll root(ll x)
    {
        if (parent[x] < 0)
            return x;
        return root(parent[x]);
    }
    bool unite(ll x, ll y)
    {
        x = root(x);
        y = root(y);
        history.push({x, parent[x]});
        history.push({y, parent[y]});
        if (x == y)
            return false;
        if (parent[x] > parent[y])
            swap(x, y);
        parent[x] += parent[y];
        parent[y] = x;
        return true;
    }
    ll same(ll x, ll y)
    {
        return root(x) == root(y);
    }
    ll size(ll x)
    {
        return (-parent[root(x)]);
    }
    void undo()
    {
        parent[history.top().first] = history.top().second;
        history.pop();
        parent[history.top().first] = history.top().second;
        history.pop();
    }
    void snapshot()
    {
        while (!history.empty())
            history.pop();
    }
    void rollback()
    {
        while (!history.empty())
        {
            undo();
        }
    }
};

// Math.
struct NCR
{
    ll max = 510000, mod = 0;
    vector<ll> fact, inv, inv_fact;
    NCR(ll mod = 998244353)
    {
        this->mod = mod;
        fact.resize(max);
        inv.resize(max);
        inv_fact.resize(max);
        fact[0] = 1;
        fact[1] = 1;
        inv[0] = 1;
        inv[1] = 1;
        inv_fact[0] = 1;
        inv_fact[1] = 1;
        for (ll i = 2; i < max; i++)
        {
            fact[i] = fact[i - 1] * i % mod;
            inv[i] = mod - inv[mod % i] * (mod / i) % mod;
            inv_fact[i] = inv_fact[i - 1] * inv[i] % mod;
        }
    }
    ll nCr(ll n, ll r)
    {
        ll x = fact[n];
        ll y = inv_fact[n - r];
        ll z = inv_fact[r];
        if (n < r)
            return 0;
        if (n < 0 || r < 0)
            return 0;
        return x * ((y * z) % mod) % mod;
    }
};

// Graph.
struct Edge
{
    ll to;
    ll cost;
    Edge(ll to, ll cost) : to(to), cost(cost) {}
    bool operator>(const Edge &e) const
    {
        return cost > e.cost;
    }
};
struct Graph
{
    vector<vector<Edge>> g;
    Graph(ll n)
    {
        g.resize(n);
    }
    ll size()
    {
        return g.size();
    }
    void add_edge(ll from, ll to, ll cost = 1)
    {
        g[from].push_back(Edge(to, cost));
        g[to].push_back(Edge(from, cost));
    }
    void add_directed_edge(ll from, ll to, ll cost = 1)
    {
        g[from].push_back(Edge(to, cost));
    }
    pair<ll, vector<ll>> tree_diameter()
    {
        function<pair<ll, ll>(ll, ll)> dfs = [&](ll v, ll p) -> pair<ll, ll>
        {
            pair<ll, ll> res = {0, v};
            for (auto e : g[v])
            {
                if (e.to == p)
                    continue;
                auto [d, u] = dfs(e.to, v);
                d += e.cost;
                if (res.first < d)
                {
                    res.first = d;
                    res.second = u;
                }
            }
            return res;
        };
        auto [d, u] = dfs(0, -1);
        auto [d2, v] = dfs(u, -1);
        vector<ll> path;
        function<void(ll, ll, ll)> get_path = [&](ll v, ll p, ll u)
        {
            if (v == u)
            {
                path.push_back(v);
                return;
            }
            for (auto e : g[v])
            {
                if (e.to == p)
                    continue;
                get_path(e.to, v, u);
                if (!path.empty())
                {
                    path.push_back(v);
                    return;
                }
            }
        };
        get_path(u, -1, v);
        return {d2, path};
    }
    vector<Edge> &operator[](ll v)
    {
        return g[v];
    }
};
struct Dijkstra
{
    vector<ll> dist;
    vector<ll> prev;

    // dijkstraを構築
    Dijkstra(Graph &g, ll s)
    {
        ll n = g.size();
        dist.assign(n, INF);
        prev.assign(n, -1);
        priority_queue<pair<ll, ll>, vector<pair<ll, ll>>, greater<pair<ll, ll>>> pq;
        dist[s] = 0;
        pq.emplace(dist[s], s);
        while (!pq.empty())
        {
            auto p = pq.top();
            pq.pop();
            ll v = p.second;
            if (dist[v] < p.first)
                continue;
            for (auto &e : g[v])
            {
                if (dist[e.to] > dist[v] + e.cost)
                {
                    dist[e.to] = dist[v] + e.cost;
                    prev[e.to] = v;
                    pq.emplace(dist[e.to], e.to);
                }
            }
        }
    }

    // 最小コストを求める
    ll get_cost(ll to)
    {
        return dist[to];
    }

    // 最短経路を求める
    vector<ll> get_path(ll to)
    {
        vector<ll> get_path;
        for (ll i = to; i != -1; i = prev[i])
        {
            get_path.push_back(i);
        }
        reverse(get_path.begin(), get_path.end());
        return get_path;
    }

    // 到達可能か調べる
    bool cango(ll to)
    {
        return (dist[to] != INF);
    }
};
struct Bellman_ford
{
    vector<ll> dist;
    vector<ll> prev;
    ll start;
    ll size;
    bool cl = false;

    // bellman_fordを構築
    Bellman_ford(Graph &g, ll s)
    {
        start = s;
        ll n = g.size();
        size = n;
        dist.assign(n, INF);
        prev.assign(n, -1);
        vector<ll> counts(n);
        vector<bool> inqueue(n);

        queue<ll> q;
        dist[s] = 0;
        q.push(s);
        inqueue[s] = true;

        while (!q.empty())
        {
            ll from = q.front();
            q.pop();
            inqueue[from] = false;

            for (auto &e : g[from])
            {
                ll d = dist[from] + e.cost;
                if (d < dist[e.to])
                {
                    dist[e.to] = d;
                    prev[e.to] = from;
                    if (!inqueue[e.to])
                    {
                        q.push(e.to);
                        inqueue[e.to] = true;
                        ++counts[e.to];

                        if (n < counts[e.to])
                            cl = true;
                    }
                }
            }
        }
    }

    // 最小コストを求める
    ll get_cost(ll to)
    {
        return dist[to];
    }

    // 最短経路を求める
    vector<ll> get_path(ll to)
    {
        vector<ll> path;
        if (dist[to] != INF)
        {
            for (ll i = to; i != -1; i = prev[i])
            {
                path.push_back(i);
            }
            reverse(path.begin(), path.end());
        }
        return path;
    }

    // 到達可能か調べる
    bool cango(ll to)
    {
        return (dist[to] != INF);
    }

    // 閉路の有無を調べる
    bool closed()
    {
        return cl;
    }
};
struct Warshall_floyd
{
    vector<vector<ll>> d;
    vector<vector<ll>> next;
    bool cl = false;

    // warshall_floydを構築
    Warshall_floyd(Graph &g)
    {
        ll n = g.size();
        d.resize(n, vector<ll>(n, INF));
        next.resize(n, vector<ll>(n, -1));

        for (ll i = 0; i < n; i++)
        {
            d[i][i] = 0;
        }

        for (ll i = 0; i < n; i++)
        {
            for (auto e : g[i])
            {
                d[i][e.to] = e.cost;
                next[i][e.to] = e.to;
            }
        }

        for (ll k = 0; k < n; ++k)
        {
            for (ll i = 0; i < n; ++i)
            {
                for (ll j = 0; j < n; ++j)
                {
                    if (d[i][j] > d[i][k] + d[k][j])
                    {
                        d[i][j] = d[i][k] + d[k][j];
                        next[i][j] = next[i][k];
                    }
                }
            }
        }

        for (ll i = 0; i < n; i++)
        {
            if (d[i][i] < 0)
            {
                cl = true;
                break;
            }
        }
    }

    // 最小コストを求める
    ll get_cost(ll from, ll to)
    {
        return d[from][to];
    }

    // 最短経路を求める
    vector<ll> get_path(ll from, ll to)
    {
        if (d[from][to] == INF)
            return {};
        vector<ll> path;
        for (ll at = from; at != to; at = next[at][to])
        {
            if (at == -1)
                return {};
            path.push_back(at);
        }
        path.push_back(to);
        return path;
    }

    // 到達可能か調べる
    bool cango(ll from, ll to)
    {
        return d[from][to] != INF;
    }

    // 負の閉路の有無を調べる
    bool closed()
    {
        return cl;
    }
};

// others.
struct Timer
{
    Timer(ll ms) : duration(ms), start_time(steady_clock::now()) {}
    bool isTimeOver() const
    {
        auto current_time = steady_clock::now();
        auto elapsed = duration_cast<milliseconds>(current_time - start_time).count();
        return elapsed >= duration;
    }
    ll duration;
    steady_clock::time_point start_time;
};

class FunctionalGraph
{
private:
    const ll V;
    ll loop_id;
    vector<ll> nx;
    void make_loop(const ll st, ll nw, vector<ll> &vec)
    {
        while (nx[nw] != st)
        {
            vec.push_back(nx[nw]);
            visit[nx[nw]] = loop_id;
            nw = nx[nw];
        }
    }
    ll dfs(ll u, vector<ll> &vec)
    {
        visit[u] = -loop_id;
        ll v = nx[u];
        if (visit[v] == -loop_id)
        {
            vec.push_back(u), vec.push_back(v);
            visit[u] = visit[v] = loop_id;
            make_loop(u, v, vec);
            loop_id++;
            return 0;
        }
        else if (visit[v] == 0)
        {
            const ll res = dfs(v, vec);
            if (res == 0)
                return 0;
            else
                return visit[u] = res;
        }
        return visit[u] = (visit[v] > 0 ? -visit[v] : visit[v]);
    }
    void make_graph()
    {
        graph.resize(V);
        for (ll i = 0; i < V; i++)
        {
            if (visit[i] < 0)
                graph[nx[i]].push_back(i);
        }
    }

public:
    vector<ll> visit;
    vector<vector<ll>> loop, graph;
    FunctionalGraph(const ll node_size)
        : V(node_size), loop_id(1), nx(V, 0), visit(V, 0) {}
    void add_edge(ll u, ll v)
    {
        nx[u] = v;
        if (u == nx[u])
            visit[u] = loop_id++, loop.push_back({u});
    }
    void solve()
    {
        for (ll i = 0; i < V; i++)
        {
            if (visit[i] == 0)
            {
                vector<ll> vec;
                dfs(i, vec);
                if (!vec.empty())
                    loop.push_back(move(vec));
            }
        }
    }
};

ll non_mod_ncr(ll n, ll r)
{
    ll res = 1;
    for (ll i = 0; i < r; i++)
    {
        res *= (n - i);
    }
    for (ll i = 0; i < r; i++)
    {
        res /= (i + 1);
    }
    return res;
}

// 前計算O(N log N)、クエリO(log N)
struct FastTruthFactorizes
{
public:
    ll N_MAX;
    vector<ll> spf;
    FastTruthFactorizes(ll N_MAX)
    {
        this->N_MAX = N_MAX;
        spf.resize(N_MAX);
        rep(i, N_MAX) spf[i] = i;
        // 調和級数の和でO(N log N).
        for (ll p = 2; p * p <= N_MAX; p++)
        {
            for (int i = p; i < N_MAX; i += p)
            {
                if (spf[i] == i)
                    spf[i] = p;
            }
        }
    }
    // 素因数分解するO(log N).
    map<ll, ll> factorize(ll n)
    {
        map<ll, ll> mp;
        while (n != 1)
        {
            ll p = spf[n];
            mp[p]++;
            n /= p;
        }
        return mp;
    }
    // 約数を列挙するO(log N).
    vector<ll> calcDevisors(ll n)
    {
        vector<ll> Y;
        auto mp = factorize(n);
        vector<pair<ll, ll>> V;
        for (auto pa : mp)
        {
            V.push_back(pa);
        }
        dfs(0, 1, Y, V);
        return Y;
    }

private:
    void dfs(ll cur_idx, ll cur_val, vector<ll> &Y, vector<pair<ll, ll>> &mp)
    {
        ll N = mp.size();
        if (cur_idx == N)
        {
            Y.push_back(cur_val);
            return;
        }
        ll v = mp[cur_idx].first;
        ll c = mp[cur_idx].second;
        ll mul = 1;
        rep(p, c + 1)
        {
            dfs(cur_idx + 1, cur_val * mul, Y, mp);
            mul *= v;
        }
        return;
    }
};

// 素数判定O(1).
bool millerRabin(ll num)
{
    if (num == 1)
        return false;
    if (num == 2)
        return true;
    if (num % 2 == 0)
        return false;
    ll d = num - 1;
    ll s = 0;
    while (d % 2 == 0)
    {
        d /= 2;
        s++;
    }
    vector<ll> primes;
    if (num < 4759123141LL)
    {
        primes = {2, 7, 61};
    }
    else
    {
        primes = {2, 325, 9375, 28178, 450775, 9780504, 1795265022};
    }
    for (auto &p : primes)
    {
        if (p >= num)
            return true;
        ll t, x = pow_mod<lll>(p, d, num);
        if (x != 1)
        {
            for (t = 0; t < s; t++)
            {
                if (x == num - 1)
                    break;
                x = lll(x) * x % num;
            }
            if (t == s)
                return false;
        }
    }
    return true;
}

// エラトステネスの篩O(N log log N).
vector<ll> eratosthenes(ll target)
{
    ll limit = sqrtl(target) + 1;
    vector<bool> primes(target + 1, true);
    primes[0] = primes[1] = false;

    for (ll n = 2; n <= limit; ++n)
    {
        if (primes[n])
        {
            for (ll multiple = n * 2; multiple <= target; multiple += n)
            {
                primes[multiple] = false;
            }
        }
    }

    vector<ll> result;
    for (ll i = 0; i <= target; ++i)
    {
        if (primes[i])
        {
            result.push_back(i);
        }
    }

    return result;
}

template <typename T>
bool next_combination(const T first, const T last, int k)
{
    const T subset = first + k;
    if (first == last || first == subset || last == subset)
    {
        return false;
    }
    T src = subset;
    while (first != src)
    {
        src--;
        if (*src < *(last - 1))
        {
            T dest = subset;
            while (*src >= *dest)
            {
                dest++;
            }
            iter_swap(src, dest);
            rotate(src + 1, dest + 1, last);
            rotate(subset, subset + (last - dest) - 1, last);
            return true;
        }
    }
    rotate(first, subset, last);
    return false;
}

int main()
{
    return 0;
}