#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#if __has_include(<bits/stdc++.h>)
#include <bits/stdc++.h>
using namespace std;
#endif
#if __has_include(<boost/multiprecision/cpp_int.hpp>)
#include <boost/multiprecision/cpp_int.hpp>
using namespace boost::multiprecision;
#endif
#if __has_include(<atcoder/all>)
#include <atcoder/all>
using namespace atcoder;
using mint = modint998244353;
using MINT = modint1000000007;
#endif
#if __has_include("./cpp-dump/cpp-dump.hpp")
#include "./cpp-dump/cpp-dump.hpp"
#endif
#include <chrono>
#include <random>
using namespace chrono;
#define rep(i, n) for (ll i = 0; i < (int)(n); ++i)
#define rep1(i, n) for (ll i = 1; i <= (n); ++i)
#define rrep(i, n) for (ll i = n; i > 0; --i)
#define bitrep(i, n) for (ll i = 0; i < (1 << n); ++i)
#define all(a) (a).begin(), (a).end()
#define yesNo(b) ((b) ? "Yes" : "No")
using ll = long long;
using ull = unsigned long long;
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

// Math.
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
void recursive_comb(vector<ll> indexes, ll s, ll rest, function<void(vector<ll>)> f)
{
    if (rest == 0)
    {
        f(indexes);
    }
    else
    {
        if (s < 0)
            return;
        recursive_comb(indexes, s - 1, rest, f);
        indexes[rest - 1] = s;
        recursive_comb(indexes, s - 1, rest - 1, f);
    }
}
void foreach_comb(ll n, ll k, function<void(vector<ll>)> f)
{
    vector<ll> indexes(k);
    recursive_comb(indexes, n - 1, k, f);
}
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
ll repeated_squaring(ll x, ll y, ll z = smallMOD)
{
    ll ans = 1;
    bitset<64> bits(y);
    string bit_str = bits.to_string();
    reverse(all(bit_str));
    rep(i, 64)
    {
        if (bit_str[i] == '1')
            ans = ans * x % z;
        x = x * x % z;
    }
    return ans;
}

// String.
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
void split(vector<string> &elems, const string &s, char delim)
{
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim))
    {
        elems.push_back(item);
    }
}
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

// algorithms.
pair<pair<ll, ll>, pair<ll, ll>> maxAndMinSubarraySum(const vector<ll> &nums)
{
    ll maxSum = nums[0];
    ll minSum = nums[0];
    ll maxStart = 0;
    ll maxEnd = 0;
    ll minStart = 0;
    ll minEnd = 0;
    ll currentMaxSum = nums[0];
    ll currentMinSum = nums[0];
    ll tempMaxStart = 0;
    ll tempMinStart = 0;

    for (ll i = 1; i < nums.size(); ++i)
    {
        if (currentMaxSum < 0)
        {
            currentMaxSum = nums[i];
            tempMaxStart = i;
        }
        else
        {
            currentMaxSum += nums[i];
        }
        if (currentMaxSum > maxSum)
        {
            maxSum = currentMaxSum;
            maxStart = tempMaxStart;
            maxEnd = i;
        }
        if (currentMinSum > 0)
        {
            currentMinSum = nums[i];
            tempMinStart = i;
        }
        else
        {
            currentMinSum += nums[i];
        }
        if (currentMinSum < minSum)
        {
            minSum = currentMinSum;
            minStart = tempMinStart;
            minEnd = i;
        }
    }
    return make_pair(make_pair(maxStart, maxEnd), make_pair(minStart, minEnd));
}

// array.
vector<string> RotateVectorString(vector<string> A, bool clockwise = false, char leading = 0)
{
    ll N = A.size();
    if (N == 0)
        return A;
    ll M = 0;
    rep(i, N)
    {
        M = max(M, (ll)A[i].size());
    }
    vector<string> B(M, string(N, leading));
    rep(i, N)
    {
        rep(j, A[i].size())
        {
            if (clockwise)
            {
                B[j][N - 1 - i] = A[i][j];
            }
            else
            {
                B[M - 1 - j][i] = A[i][j];
            }
        }
    }
    return B;
}

// struct.
// Data structure.
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
    ll query(ll x1, ll y1, ll x2, ll y2)
    {
        return data[x2][y2] - data[x1][y2] - data[x2][y1] + data[x1][y1];
    }
};
struct CumulativeSumND
{
public:
    vector<ll> sum;
    ll N;
    ll d;
    CumulativeSumND(ll d, ll N, vector<ll> &data)
    {
        this->d = d;
        this->N = N;
        ll cumSize = 1;
        ll size = 1;
        ll bitSize = 1;
        for (ll i = 0; i < d; i++)
        {
            cumSize *= N + 1;
            size *= N;
            bitSize *= 2;
            assert(cumSize <= 1e8);
            assert(size <= 1e8);
            assert(bitSize <= 1e8);
        }
        assert(data.size() == size);
        sum.resize(cumSize);
        build(data);
    }
    // (X1,Y1,Z1,W1,...,X2,Y2,Z2,W2,...)
    ll query(vector<ll> q)
    {
        assert(q.size() == d * 2);
        vector<ll> LPos(d + 1, 0);
        vector<ll> RPos(d + 1, 0);
        ll res = 0;
        for (ll i = 0; i < d; i++)
        {
            LPos[i + 1] = q[i] - 1;
            RPos[i + 1] = q[i + d];
        }
        bitrep(i, d)
        {
            vector<ll> pos(d + 1, 0);
            ll cnt = 0;
            rep(j, d)
            {
                if (i & (1 << j))
                {
                    pos[j + 1] = RPos[j + 1];
                }
                else
                {
                    pos[j + 1] = LPos[j + 1];
                    cnt++;
                }
            }
            if (cnt % 2)
            {
                res -= sum[getPos(pos, 1)];
            }
            else
            {
                res += sum[getPos(pos, 1)];
            }
        }
        return res;
    }

private:
    // 0-indexed.
    ll getPos(vector<ll> &pos, bool isZeroIndex)
    {
        assert(pos.size() == d + 1);
        vector<ll> A(d, 1);
        for (ll i = 0; i < d - 1; i++)
        {
            A[i + 1] = A[i] * (N + isZeroIndex);
        }
        reverse(all(A));
        ll res = 0;
        for (ll i = 1; i < d + 1; i++)
        {
            res += A[i - 1] * pos[i];
        }
        return res;
    }
    void build(vector<ll> &data)
    {
        {
            // 0-indexed→1-indexed.
            vector<ll> loop(d + 1, 0);
            while (loop[0] != 1)
            {
                bool isZero = false;
                for (ll i = 1; i < d + 1; i++)
                {
                    if (loop[i] == 0)
                    {
                        isZero = true;
                        break;
                    }
                }
                if (!isZero)
                {
                    ll sumPos = getPos(loop, 1);
                    for (ll i = 1; i < d + 1; i++)
                    {
                        loop[i]--;
                    }
                    ll dataPos = getPos(loop, 0);
                    for (ll i = 1; i < d + 1; i++)
                    {
                        loop[i]++;
                    }
                    sum[sumPos] = data[dataPos];
                }
                loop[d]++;
                // 繰り上がり処理.
                for (ll i = d; i > 0; i--)
                {
                    if (loop[i] == N + 1)
                    {
                        loop[i] = 0;
                        loop[i - 1]++;
                    }
                }
            }
        }
        // calc sum.
        {
            vector<ll> loop(d + 1, 0);
            while (loop[0] != d)
            {
                ll p1 = getPos(loop, 1);
                loop[loop[0] + 1]++;
                ll p2 = getPos(loop, 1);
                sum[p2] += sum[p1];
                loop[loop[0] + 1]--;
                loop[d]++;
                // 繰り上がり処理.
                for (ll i = d; i > 0; i--)
                {
                    if (i == loop[0] + 1)
                    {
                        // N-1回.
                        if (loop[i] == N)
                        {
                            loop[i] = 0;
                            loop[i - 1]++;
                        }
                    }
                    else
                    {
                        // N回.
                        if (loop[i] == N + 1)
                        {
                            loop[i] = 0;
                            loop[i - 1]++;
                        }
                    }
                }
            }
        }
    }
};
struct BitVector
{
public:
    vector<int> data, cum;
    BitVector(vector<int> &source)
    {
        if (source.size() != 0)
        {
            data.resize(source.size(), 0);
            cum.resize(source.size(), 0);
            cum[0] = source[0];
            for (int i = 0; i < (int)source.size(); i++)
            {
                data[i] = source[i];
                assert(source[i] == 0 || source[i] == 1);
                if (i != 0)
                {
                    cum[i] = cum[i - 1] + source[i];
                }
            }
        }
    }
    int size()
    {
        return (int)data.size();
    }
    int get(int i)
    {
        return data[i];
    }
    int rank1(int i)
    {
        if (i <= 0)
        {
            return 0;
        }
        i = min(i, (int)data.size());
        return cum[i - 1];
    }
    int rank0(int i)
    {
        if (i <= 0)
        {
            return 0;
        }
        i = min(i, (int)data.size());
        return i - rank1(i);
    }
    int rank0_all()
    {
        return rank0(data.size());
    }
    int rank1_all()
    {
        return rank1(data.size());
    }
    int select0(int k)
    {
        // K個目の0のindexを返す.
        if (k <= 0 || k > rank0_all())
        {
            return -1;
        }
        int l = 0, r = data.size();
        while (r - l > 1)
        {
            int mid = (l + r) / 2;
            if (rank0(mid) < k)
            {
                l = mid;
            }
            else
            {
                r = mid;
            }
        }
        return l;
    }
    int select1(int k)
    {
        // K個目の1のindexを返す.
        if (k <= 0 || k > rank1_all())
        {
            return -1;
        }
        int l = 0, r = data.size();
        while (r - l > 1)
        {
            int mid = (l + r) / 2;
            if (rank1(mid) < k)
            {
                l = mid;
            }
            else
            {
                r = mid;
            }
        }
        return l;
    }
};
struct UnionFind
{
    vector<ll> parent;
    vector<ll> parentSize;
    vector<ll> scoresA;
    vector<ll> scoresB;
    UnionFind(ll N, vector<ll> scoresA = {}, vector<ll> scoresB = {})
    {
        this->scoresA = scoresA;
        this->scoresB = scoresB;
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
    void decScoreA(ll x, ll u)
    {
        scoresA[x] -= u;
    }
    void decScoreB(ll x, ll u)
    {
        scoresB[x] -= u;
    }
    void unite(ll x, ll y)
    {
        x = root(x);
        y = root(y);
        if (x == y)
            return;
        if (parentSize[x] <= parentSize[y])
        {
            parent[x] = y;
            parentSize[y] += parentSize[x];
            scoresA[y] += scoresA[x];
            scoresB[y] += scoresB[x];
        }
        else
        {
            parent[y] = x;
            parentSize[x] += parentSize[y];
            scoresA[x] += scoresA[y];
            scoresB[x] += scoresB[y];
        }
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

// String.
struct Suffix_array
{
    vector<ll> sa, rank, tmp;
    ll n;
    string base;

    // suffix_arrayを構築
    Suffix_array(const string s)
    {
        n = s.size();
        base = s;
        sa.resize(n);
        rank.resize(n);
        tmp.resize(n);

        for (ll i = 0; i < n; i++)
        {
            sa[i] = i;
            rank[i] = s[i];
        }

        for (ll k = 1; k < n; k *= 2)
        {
            auto compare_sa = [&](ll i, ll j)
            {
                if (rank[i] != rank[j])
                    return rank[i] < rank[j];
                ll ri = (i + k < n) ? rank[i + k] : -1;
                ll rj = (j + k < n) ? rank[j + k] : -1;
                return ri < rj;
            };
            sort(sa.begin(), sa.end(), compare_sa);

            tmp[sa[0]] = 0;
            for (ll i = 1; i < n; i++)
            {
                tmp[sa[i]] = tmp[sa[i - 1]] + (compare_sa(sa[i - 1], sa[i]) ? 1 : 0);
            }
            rank = tmp;
        }
    }

    // 部分文字列の個数を求める
    ll get_counts(string t)
    {
        string u = t + "彁";
        ll num1, num2;
        {
            ll m = t.size();
            ll ng = -1, ok = n;
            while (ok - ng > 1)
            {
                ll mid = (ng + ok) / 2;
                ll l = sa[mid];
                if (base.substr(l, m) >= t)
                {
                    ok = mid;
                }
                else
                {
                    ng = mid;
                }
            }
            num1 = ok;
        }
        {
            ll m = u.size();
            ll ng = -1, ok = n;
            while (ok - ng > 1)
            {
                ll mid = (ng + ok) / 2;
                ll l = sa[mid];
                if (base.substr(l, m) >= u)
                {
                    ok = mid;
                }
                else
                {
                    ng = mid;
                }
            }
            num2 = ok;
        }
        return num2 - num1;
    }

    // make lcp array
    vector<ll> make_lcp()
    {
        vector<ll> lcp(n);
        for (ll i = 0; i < n; i++)
        {
            rank[sa[i]] = i;
        }
        ll h = 0;
        lcp[0] = 0;
        for (ll i = 0; i < n; i++)
        {
            if (rank[i] == n - 1)
            {
                h = 0;
                continue;
            }
            ll j = sa[rank[i] + 1];
            while (i + h < n && j + h < n && base[i + h] == base[j + h])
            {
                h++;
            }
            lcp[rank[i]] = h;
            if (h > 0)
            {
                h--;
            }
        }
        return lcp;
    }
};

// others.
class Timer
{
public:
    Timer(ll ms) : duration(ms), start_time(steady_clock::now()) {}
    bool isTimeOver() const
    {
        auto current_time = steady_clock::now();
        auto elapsed = duration_cast<milliseconds>(current_time - start_time).count();
        return elapsed >= duration;
    }

private:
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

void print2DArr(vector<vector<ll>> &arr)
{
    for (auto &a : arr)
    {
        for (auto &b : a)
        {
            cout << b << " ";
        }
        cout << endl;
    }
    cout << endl;
}

string encode(ll n)
{
    string res = "";
    while (n)
    {
        n--;
        res = ALPHABET[n % 26] + res;
        n /= 26;
    }
    return res;
}

ll decode(string s)
{
    ll res = 0;
    for (char c : s)
    {
        res = res * 26 + c - 'A' + 1;
    }
    return res;
}

ll digit_sum(ll n)
{
    ll sum = 0;
    while (n > 0)
    {
        sum += n % 10;
        n /= 10;
    }
    return sum;
}

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

int rand_int(int a, int b)
{
    return a + rand() % (b - a + 1);
}

int main()
{
    return 0;
}
