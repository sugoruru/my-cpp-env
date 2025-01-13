#include <bits/stdc++.h>
using namespace std;

#define rep(i, l, r) for (int i = (int)(l); i < (int)(r); i++)
#define ll long long

vector<int> di = {-1, 0, 1, 0, -1, 1, -1, 1};
vector<int> dj = {0, -1, 0, 1, 1, -1, -1, 1};

int main()
{
    int H, W;
    cin >> H >> W;
    int A, B;
    cin >> A >> B;
    vector<int> cnt = {0, 0};
    vector<string> S(H);
    rep(i, 0, H) cin >> S[i];
    rep(i, 0, H)
    {
        rep(j, 0, W)
        {
            if (S[i][j] == '.')
                cnt[0]++;
            else
                cnt[1]++;
        }
    }
    int a = cnt[0], b = cnt[1];
    int g = gcd(a, b);
    a /= g;
    b /= g;
    cout << a << " " << b << endl;
    if (a != A or b != B)
    {
        cout << "No counts" << endl;
        return 0;
    }
    bool ok = true;
    rep(i, 0, H) rep(j, 0, W)
    {
        bool f = false;
        rep(k, 0, 4)
        {
            int ni = i + di[k], nj = j + dj[k];
            ni = (ni + H) % H;
            nj = (nj + W) % W;
            if (S[ni][nj] != S[i][j])
                f = true;
        }
        if (!f)
            ok = false;
    }
    if (!ok)
        cout << "No error" << endl;
    else
        cout << "Yes" << endl;
}