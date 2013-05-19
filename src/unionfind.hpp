#ifndef UNIONFIND_H
#define UNIONFIND_H

class UnionFindSet
{
private:
    const int max;
    int *parent;
public:
    UnionFindSet(int max);
    ~UnionFindSet();
    int findSet(int x);
    void unionSet(int x, int y);
};

#endif
