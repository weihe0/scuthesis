#include <opencv2/core/core.hpp>
#include "unionfind.hpp"
using namespace std;
UnionFindSet::UnionFindSet(int max) : max(max)
{
    parent = new int[max + 1];
    for (int i = 0; i <= max; i++)
    {
	parent[i] = 0;
    }
}

UnionFindSet::~UnionFindSet()
{
    delete[] parent;
}

int UnionFindSet::findSet(int x)
{
    CV_Assert(x >= 1 && x <= max);
    if (parent[x] == 0)
    {
	return x;
    }
    else
    {
	parent[x] = findSet(parent[x]);
	return parent[x];
    }
}

void UnionFindSet::unionSet(int x, int y)
{
    CV_Assert(x >= 1 && x <= max);
    CV_Assert(y >= 1 && y <= max);
    int p = findSet(x);
    int q = findSet(y);
    if (p != q)
    {
	parent[p] = q;
    }
}
